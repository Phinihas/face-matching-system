CREATE TABLE IF NOT EXISTS pii_status (
    request_id VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status CHAR(1) NOT NULL DEFAULT 'p',
    processed_count INT DEFAULT 0,
    total_count INT DEFAULT 0,
    request_type VARCHAR(50) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (request_id, type)
);

CREATE TABLE IF NOT EXISTS pii_results (
    id VARCHAR(255) NOT NULL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    result JSON,
    filepath VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_request_id (request_id)
);
