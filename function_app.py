import azure.functions as func
import logging
import pyodbc
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import create_engine, pool, event
from sqlalchemy.engine import Engine

# Initialize Azure Functions app with function-level authentication
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# ============================================
# ML Model Singleton Cache
# ============================================
# Global cache to store loaded ResNet models across function invocations
# This avoids reloading models on every task, significantly improving performance
_model_cache = {}
_model_cache_lock = threading.Lock()

def get_cached_model(model_type: str):
    """
    Retrieve a cached ResNet model or load it if not already cached.
    Thread-safe singleton pattern to ensure models are loaded only once.
    
    Args:
        model_type (str): Either 'ResNet18' or 'ResNet50'
    
    Returns:
        torch.nn.Module: The requested ResNet model in evaluation mode
    
    Raises:
        ValueError: If model_type is not supported
    """
    # Check if model is already cached (fast path without lock)
    if model_type in _model_cache:
        logging.debug(f'Using cached {model_type} model')
        return _model_cache[model_type]
    
    # Acquire lock for thread-safe model loading
    with _model_cache_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if model_type in _model_cache:
            logging.debug(f'Using cached {model_type} model (loaded by another thread)')
            return _model_cache[model_type]
        
        # Load PyTorch and models (only import when needed)
        import torch
        import torchvision.models as models
        
        logging.info(f'Loading {model_type} model for the first time...')
        
        # Load the appropriate model architecture
        if model_type == 'ResNet18':
            model = models.resnet18(pretrained=False)
        elif model_type == 'ResNet50':
            model = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Set to evaluation mode (disables dropout, batch norm updates)
        model.eval()
        
        # Cache the model for future use
        _model_cache[model_type] = model
        logging.info(f'{model_type} model loaded and cached successfully')
        
        return model

# ============================================
# Database Connection Pool
# ============================================
# SQLAlchemy connection pool for efficient database access
# Reuses connections instead of creating new ones for each query
_db_engine = None
_db_engine_lock = threading.Lock()

def _get_connection_string():
    """Build the full ODBC connection string for SQL Server"""
    base_connection_string = os.environ.get('SQL_CONNECTION_STRING')
    if not base_connection_string:
        raise ValueError("SQL_CONNECTION_STRING environment variable not set")
    return f"DRIVER={{ODBC Driver 18 for SQL Server}};{base_connection_string}"

def _initialize_db_engine():
    """
    Initialize SQLAlchemy engine with connection pooling.
    Thread-safe singleton pattern ensures only one pool is created.
    """
    global _db_engine
    
    if _db_engine is not None:
        return _db_engine
    
    with _db_engine_lock:
        # Double-check after acquiring lock
        if _db_engine is not None:
            return _db_engine
        
        connection_string = _get_connection_string()
        # URL-encode the connection string for SQLAlchemy
        connection_url = f"mssql+pyodbc:///?odbc_connect={connection_string}"
        
        # Create engine with connection pool configuration
        _db_engine = create_engine(
            connection_url,
            poolclass=pool.QueuePool,
            pool_size=8,              # Keep 8 connections open
            max_overflow=24,          # Allow up to 32 total connections (8 + 24)
            pool_timeout=30,          # Wait up to 30s for a connection
            pool_recycle=3600,        # Recycle connections after 1 hour
            pool_pre_ping=True,       # Verify connection health before using
            echo=False                # Disable SQL query logging
        )
        
        logging.info('Database connection pool initialized (size=8, max=32)')
        return _db_engine

@contextmanager
def get_db_connection():
    """
    Context manager to get a database connection from the pool.
    Automatically handles connection checkout and return.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ...")
    """
    engine = _initialize_db_engine()
    
    # Get raw pyodbc connection from SQLAlchemy pool
    connection = engine.raw_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        # Return connection to pool (not closing it)
        connection.close()

# ============================================
# Function 1: CreateTask (HTTP Trigger)
# ============================================
@app.route(route="create_task", methods=["POST"])
def create_task(req: func.HttpRequest) -> func.HttpResponse:
    """
    Function 1: CreateTask
    - Triggered by HTTP POST request
    - Accepts model type (ResNet18 or ResNet50)
    - Inserts a new task into InferenceTasks table
    """
    logging.info('CreateTask function triggered via HTTP.')

    try:
        # Extract and validate the requested model type from request body
        req_body = req.get_json()
        model_type = req_body.get('model_type', 'ResNet18')

        # Define supported model types
        if model_type not in ['ResNet18', 'ResNet50']:
            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid model_type. Must be 'ResNet18' or 'ResNet50'."
                }),
                status_code=400,
                mimetype="application/json"
            )

        # Insert new task record into database using connection pool
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create task with 'Pending' status and F1 timestamp
            insert_query = """
                INSERT INTO InferenceTasks (ModelToUse, Status, TimestampF1)
                VALUES (?, ?, ?)
            """
            cursor.execute(insert_query, model_type, 'Pending', datetime.now(timezone.utc))
            conn.commit()

            # Retrieve the auto-generated TaskID
            cursor.execute("SELECT @@IDENTITY AS TaskID")
            task_id = cursor.fetchone()[0]

            logging.info(f'Task created successfully. TaskID: {task_id}, Model: {model_type}')

            # Return success response with task details
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "task_id": int(task_id),
                    "model_type": model_type,
                    "message": "Task created successfully and queued for processing."
                }),
                status_code=201,
                mimetype="application/json"
            )

    except ValueError as ve:
        # Handle malformed JSON in request body
        logging.error(f'Invalid request body: {str(ve)}')
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body."}),
            status_code=400,
            mimetype="application/json"
        )

    except pyodbc.Error as db_error:
        # Handle database connection or query errors
        logging.error(f'Database error: {str(db_error)}')
        return func.HttpResponse(
            json.dumps({"error": f"Database error: {str(db_error)}"}),
            status_code=500,
            mimetype="application/json"
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logging.error(f'Unexpected error: {str(e)}')
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


# ============================================
# Function 2: ProcessTask (SQL Trigger)
# ============================================
@app.sql_trigger(
    arg_name="changes",
    table_name="InferenceTasks",
    connection_string_setting="SQL_CONNECTION_STRING"
)
def process_task(changes: str):
    """
    Function 2: ProcessTask
    - Triggered by INSERT events on InferenceTasks table
    - Loads specified ResNet model
    - Performs inference on simulated image
    - Records performance metrics to InferenceResults table
    """
    # Import PyTorch libraries here to avoid cold start overhead for F1
    import torch
    import torchvision.models as models

    # Parse incoming change events from SQL trigger
    changes_list = json.loads(changes)

    # Track statistics for logging
    tasks_to_process = []
    skipped_count = 0
    error_count = 0

    # Process each task from the trigger batch
    for change in changes_list:
        # Extract data from the item property
        item = change.get('Item') or change
        
        # Extract fields with case-insensitive key matching
        task_id = (item.get('TaskID') or item.get('taskid') or item.get('TaskId') or 
                   item.get('TASKID') or item.get('taskId'))
        model_to_use = (item.get('ModelToUse') or item.get('modelToUse') or 
                        item.get('modeltouse') or item.get('MODELTOUSE') or item.get('ModeltoUse'))
        status = (item.get('Status') or item.get('status') or 'Unknown')

        # Skip if critical fields are missing
        if not task_id or not model_to_use:
            error_count += 1
            logging.error(f'Missing required fields. TaskID: {task_id}, ModelToUse: {model_to_use}')
            continue

        # Avoid reprocessing completed or failed tasks (silent skip)
        if status in ['Completed', 'Failed']:
            skipped_count += 1
            continue

        # Add to processing queue
        tasks_to_process.append((task_id, model_to_use))

    # Log summary instead of individual entries
    logging.info(f'SQL Trigger: {len(changes_list)} total, {len(tasks_to_process)} to process, {skipped_count} skipped, {error_count} errors')

    # Process only the pending tasks
    for task_id, model_to_use in tasks_to_process:
        logging.info(f'Processing TaskID: {task_id}, Model: {model_to_use}')

        # Mark F2 function start time for performance measurement
        f2_start_time = datetime.now(timezone.utc)

        try:
            # Create a random image tensor simulating a 224x224 RGB image
            # Shape: [batch_size=1, channels=3, height=224, width=224]
            simulated_image = torch.randn(1, 3, 224, 224)

            # Get model from cache (or load if first time)
            # This significantly reduces latency by avoiding repeated model loading
            model = get_cached_model(model_to_use)

            # Run inference
            # Note: torch.no_grad() is used to disable gradient tracking
            with torch.no_grad():
                output = model(simulated_image)
            predicted_class = torch.argmax(output, dim=1).item()

            # Mark F2 function end time
            f2_end_time = datetime.now(timezone.utc)

            # Calculate total inference time in milliseconds
            execution_time_ms = int((f2_end_time - f2_start_time).total_seconds() * 1000)

            # Store inference results and update task status using connection pool
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Insert performance metrics and classification result
                # F2_ExecutionTime_ms is auto-calculated by database as a computed column
                insert_result_query = """
                    INSERT INTO InferenceResults (
                        TaskID,
                        ModelUsed,
                        F2_StartTime,
                        F2_EndTime,
                        ClassificationOutput
                    )
                    VALUES (?, ?, ?, ?, ?)
                """
                
                try:
                    cursor.execute(
                        insert_result_query,
                        int(task_id),
                        str(model_to_use),
                        f2_start_time,
                        f2_end_time,
                        str(predicted_class)
                    )
                except pyodbc.Error as db_err:
                    logging.error(f'Database insert error for TaskID {task_id}: {str(db_err)}')
                    raise

                # Mark the task as completed in the InferenceTasks table
                update_task_query = """
                    UPDATE InferenceTasks
                    SET Status = 'Completed'
                    WHERE TaskID = ?
                """
                cursor.execute(update_task_query, int(task_id))

                conn.commit()

            # Log completion with key metrics in one line
            logging.info(f'TaskID {task_id} completed: {execution_time_ms}ms, class={predicted_class}')

        except Exception as e:
            # Handle any errors during model loading or inference
            logging.error(f'Error processing TaskID {task_id}: {str(e)}')

            # Mark the task as failed in the database for tracking
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    update_task_query = """
                        UPDATE InferenceTasks
                        SET Status = 'Failed'
                        WHERE TaskID = ?
                    """
                    cursor.execute(update_task_query, task_id)
                    conn.commit()
            except Exception as db_error:
                # Log if we can't even update the failure status
                logging.error(f'Failed to update task status: {str(db_error)}')