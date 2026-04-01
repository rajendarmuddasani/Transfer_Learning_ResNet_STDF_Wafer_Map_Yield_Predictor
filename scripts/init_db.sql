-- STDF Wafer Map Yield Predictor Database Initialization

-- Products Table
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    package_type VARCHAR(20) NOT NULL,
    die_count INT NOT NULL,
    max_x INT NOT NULL,
    max_y INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Lots Table
CREATE TABLE IF NOT EXISTS lots (
    lot_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50) REFERENCES products(product_id),
    fab_site VARCHAR(20),
    start_date DATE,
    end_date DATE,
    target_yield DECIMAL(5,2),
    actual_yield DECIMAL(5,2),
    status VARCHAR(20)
);

-- Wafers Table
CREATE TABLE IF NOT EXISTS wafers (
    wafer_id VARCHAR(50) PRIMARY KEY,
    lot_id VARCHAR(50) REFERENCES lots(lot_id),
    wafer_num INT NOT NULL,
    die_count INT,
    pass_count INT,
    fail_count INT,
    stdf_path VARCHAR(500),
    wafer_map_url VARCHAR(500),
    test_completion_pct DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lot_wafer ON wafers(lot_id, wafer_num);
CREATE INDEX IF NOT EXISTS idx_test_completion ON wafers(test_completion_pct);

-- Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    wafer_id VARCHAR(50) REFERENCES wafers(wafer_id),
    model_version VARCHAR(50) NOT NULL,
    yield_pred DECIMAL(5,2) NOT NULL,
    defect_class VARCHAR(50),
    confidence DECIMAL(5,4),
    inference_time_ms INT,
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_wafer_model ON predictions(wafer_id, model_version);
CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);

-- GradCAM Heatmaps Table
CREATE TABLE IF NOT EXISTS gradcam_heatmaps (
    gradcam_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    heatmap_url VARCHAR(500),
    layer_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    architecture VARCHAR(50),
    version VARCHAR(20),
    onnx_path VARCHAR(500),
    pytorch_path VARCHAR(500),
    stage VARCHAR(20),
    accuracy DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(50)
);

-- Experiments Table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    run_id VARCHAR(100) UNIQUE,
    hyperparams JSONB,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),
    training_samples INT,
    validation_samples INT
);

-- Model Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
    metric_name VARCHAR(50),
    metric_value DECIMAL(10,6),
    step INT,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_experiment_metric ON model_metrics(experiment_id, metric_name);

-- Datasets Table
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR(50) PRIMARY KEY,
    dataset_name VARCHAR(100),
    version VARCHAR(20),
    split_type VARCHAR(10),
    sample_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    dvc_hash VARCHAR(100),
    description TEXT
);

-- WaferMap Images Table
CREATE TABLE IF NOT EXISTS wafermap_images (
    image_id BIGSERIAL PRIMARY KEY,
    wafer_id VARCHAR(50) REFERENCES wafers(wafer_id),
    dataset_id VARCHAR(50) REFERENCES datasets(dataset_id),
    image_url VARCHAR(500),
    defect_label VARCHAR(50),
    yield_actual DECIMAL(5,2),
    resolution VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_label ON wafermap_images(dataset_id, defect_label);

-- Insert sample data
INSERT INTO products (product_id, product_name, package_type, die_count, max_x, max_y) 
VALUES ('TC42x', 'TC42x Automotive MCU', 'BGA436', 850, 50, 50)
ON CONFLICT (product_id) DO NOTHING;

INSERT INTO models (model_id, model_name, architecture, version, stage, accuracy, created_by)
VALUES 
    ('resnet18-v1.2', 'ResNet-18 Transfer Learning', 'resnet18', 'v1.2', 'PRODUCTION', 0.9245, 'system'),
    ('resnet50-v2.0', 'ResNet-50 Transfer Learning', 'resnet50', 'v2.0', 'STAGING', 0.9387, 'system')
ON CONFLICT (model_id) DO NOTHING;
