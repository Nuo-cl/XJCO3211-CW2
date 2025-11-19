-- ============================================
-- Database Schema for ML Performance Evaluation Workflow
-- ============================================

-- Table 1: InferenceTasks
-- Purpose: Task queue for inference requests
CREATE TABLE InferenceTasks (
    TaskID INT IDENTITY(1,1) PRIMARY KEY,
    ModelToUse NVARCHAR(50) NOT NULL,  -- 'ResNet18' or 'ResNet50'
    Status NVARCHAR(20) NOT NULL DEFAULT 'Pending',  -- 'Pending', 'Completed', 'Failed'
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    CompletedAt DATETIME2 NULL
);

-- Table 2: InferenceResults
-- Purpose: Store performance metrics and inference results
CREATE TABLE InferenceResults (
    ResultID INT IDENTITY(1,1) PRIMARY KEY,
    TaskID INT NOT NULL,
    ModelUsed NVARCHAR(50) NOT NULL,  -- 'ResNet18' or 'ResNet50'
    F2_StartTime DATETIME2 NOT NULL,
    F2_EndTime DATETIME2 NOT NULL,
    F2_ExecutionTime_ms AS DATEDIFF(MILLISECOND, F2_StartTime, F2_EndTime) PERSISTED,  -- Computed column
    PredictedClass INT NULL,  -- Simulated classification result
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    FOREIGN KEY (TaskID) REFERENCES InferenceTasks(TaskID)
);

-- Indexes for better query performance
CREATE INDEX IX_InferenceTasks_Status ON InferenceTasks(Status);
CREATE INDEX IX_InferenceTasks_CreatedAt ON InferenceTasks(CreatedAt);
CREATE INDEX IX_InferenceResults_TaskID ON InferenceResults(TaskID);
CREATE INDEX IX_InferenceResults_ModelUsed ON InferenceResults(ModelUsed);

-- Enable Change Tracking for SQL Trigger (required for Azure Functions SQL Trigger)
ALTER DATABASE CURRENT
SET CHANGE_TRACKING = ON
(CHANGE_RETENTION = 2 DAYS, AUTO_CLEANUP = ON);

ALTER TABLE InferenceTasks
ENABLE CHANGE_TRACKING;
