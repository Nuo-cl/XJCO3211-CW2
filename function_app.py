import azure.functions as func
import logging
import pyodbc
import json
import os
from datetime import datetime, timezone

# Initialize Azure Functions app with function-level authentication
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

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

        # Retrieve database connection string from environment variables
        connection_string = os.environ.get('SQL_CONNECTION_STRING')
        if not connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Database connection string not configured."}),
                status_code=500,
                mimetype="application/json"
            )

        # Build full connection string with ODBC driver specification
        full_connection_string = f"Driver={{ODBC Driver 18 for SQL Server}};{connection_string}"

        # Insert new task record into database
        with pyodbc.connect(full_connection_string) as conn:
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

    logging.info(f'ProcessTask function triggered. Processing {len(changes_list)} task(s).')

    # Prepare database connection for result storage
    connection_string = os.environ.get('SQL_CONNECTION_STRING')
    full_connection_string = f"Driver={{ODBC Driver 18 for SQL Server}};{connection_string}"

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
            logging.error(f'Missing required fields. TaskID: {task_id}, ModelToUse: {model_to_use}')
            continue

        # Avoid reprocessing completed or failed tasks
        if status in ['Completed', 'Failed']:
            logging.warning(f'TaskID {task_id} is already {status}. Skipping.')
            continue

        logging.info(f'Processing TaskID: {task_id}, Model: {model_to_use}')

        # Mark F2 function start time for performance measurement
        f2_start_time = datetime.now(timezone.utc)

        try:
            # Create a random image tensor simulating a 224x224 RGB image
            # Shape: [batch_size=1, channels=3, height=224, width=224]
            simulated_image = torch.randn(1, 3, 224, 224)

            # Load the requested ResNet model architecture
            # Note: pretrained=False to avoid downloading weights (faster for testing)
            if model_to_use == 'ResNet18':
                model = models.resnet18(pretrained=False)
            elif model_to_use == 'ResNet50':
                model = models.resnet50(pretrained=False)
            else:
                raise ValueError(f"Unknown model type: {model_to_use}")

            # Switch to evaluation mode
            model.eval()

            # Run inference
            # Note: torch.no_grad() is used to disable gradient tracking
            with torch.no_grad():
                output = model(simulated_image)
            predicted_class = torch.argmax(output, dim=1).item()

            # Mark F2 function end time
            f2_end_time = datetime.now(timezone.utc)

            # Calculate total inference time in milliseconds
            execution_time_ms = int((f2_end_time - f2_start_time).total_seconds() * 1000)

            logging.info(f'TaskID {task_id} inference completed. Time: {execution_time_ms}ms, Predicted Class: {predicted_class}')

            # Store inference results and update task status in database
            with pyodbc.connect(full_connection_string) as conn:
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

            logging.info(f'TaskID {task_id} results saved to database.')

        except Exception as e:
            # Handle any errors during model loading or inference
            logging.error(f'Error processing TaskID {task_id}: {str(e)}')

            # Mark the task as failed in the database for tracking
            try:
                with pyodbc.connect(full_connection_string) as conn:
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