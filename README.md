# ML Inference Serverless Workflow

A serverless machine learning inference system built on Azure Functions, demonstrating production-grade performance optimization strategies for compute-intensive workloads.

## Overview

This project implements an event-driven ML inference pipeline using Azure Functions and Azure SQL Database. The system accepts ResNet18/50 classification requests via HTTP API and processes them asynchronously using SQL triggers, achieving significant performance improvements through systematic optimization.

## Architecture

```
Client → CreateTask (HTTP) → Azure SQL Database → ProcessTask (SQL Trigger) → Results
```

### Components

- **Function 1 (CreateTask)**: HTTP-triggered API endpoint for task submission
- **Function 2 (ProcessTask)**: SQL-triggered inference processor
- **Azure SQL Database**: Task queue and results storage with Change Tracking
- **ResNet Models**: PyTorch-based image classification models (ResNet18/50)

## Key Features

- **Event-Driven Architecture**: SQL Change Tracking enables automatic inference triggering
- **Asynchronous Processing**: Immediate API response with background inference execution
- **Model Support**: ResNet18 and ResNet50 pre-trained models
- **Performance Tracking**: End-to-end timing metrics (F1, F2_StartTime, F2_EndTime)

## Performance Optimizations

The system incorporates three layers of optimization:

### 1. Model Singleton Cache
- **Problem**: Repetitive model loading (50-400ms per request)
- **Solution**: Thread-safe singleton cache with persistent model instances

### 2. Database Connection Pooling
- **Problem**: Connection overhead (50-200ms per operation)
- **Solution**: SQLAlchemy connection pool (8 persistent, 32 maximum)

### 3. Concurrency Configuration
- **Problem**: Single-threaded processing, slow polling, unlimited concurrency
- **Solution**: 6 worker threads, 1.5s polling, 30 concurrent requests

## Technology Stack

**Platform**
- Azure Functions (Python 3.12) on Basic B3 plan
- Azure SQL Database (Free tier with Change Tracking)
- Azure Storage Account

**Languages & Libraries**
- Python 3.12
- PyTorch (CPU) + torchvision
- SQLAlchemy (connection pooling)
- pyodbc (SQL Server driver)

**Development Tools**
- Visual Studio Code + Azure Functions extension
- Azure Functions Core Tools v4
- Conda (environment management)

## Project Structure

```
TaskSender/
├── function_app.py              # Main application code
│   ├── Model caching logic
│   ├── Connection pooling
│   ├── CreateTask function
│   └── ProcessTask function
├── host.json                    # Runtime configuration
├── requirements.txt             # Production dependencies
├── local.settings.json.example  # Environment template
├── database_schema.sql          # Database setup script
```

## API Usage

### Create Inference Task

**Endpoint**: `POST /api/create_task`

**Request**:
```json
{
  "model_type": "ResNet18"
}
```

**Response** (HTTP 201):
```json
{
  "status": "success",
  "task_id": 123,
  "model_type": "ResNet18",
  "message": "Task created successfully and queued for processing."
}
```

**Supported Models**:
- `ResNet18`: Faster inference, lower accuracy
- `ResNet50`: Slower inference, higher accuracy

### Query Task Status

Tasks can be queried directly from the database:

```sql
-- Check task status
SELECT TaskID, ModelToUse, Status, TimestampF1
FROM InferenceTasks
WHERE TaskID = 123;

-- Get inference results
SELECT r.*, t.ModelToUse
FROM InferenceResults r
JOIN InferenceTasks t ON r.TaskID = t.TaskID
WHERE r.TaskID = 123;
```


### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SQL_CONNECTION_STRING` | Azure SQL connection string | `Server=...;Database=...;` |
| `PYTHON_THREADPOOL_THREAD_COUNT` | Worker thread count | `6` |
| `PYTHON_ENABLE_WORKER_EXTENSIONS` | Enable extensions | `1` |


## Author

Nuo Chen - XJCO3211 Distributed Systems Coursework