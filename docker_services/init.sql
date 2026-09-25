CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id VARCHAR(8) PRIMARY KEY,
    status ENUM('pending', 'processing', 'completed', 'failed') NOT NULL DEFAULT 'pending',
    folder_path VARCHAR(500) NOT NULL,
    max_workers INT NOT NULL DEFAULT 4,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    total_images INT DEFAULT 0,
    processed_images INT DEFAULT 0,
    error_message TEXT NULL,
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);