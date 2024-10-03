-- Create the ASTAPM database
CREATE DATABASE IF NOT EXISTS astapm;
CREATE DATABASE IF NOT EXISTS asrs_raw;
CREATE DATABASE IF NOT EXISTS ntsb_raw;


-- Switch to the ASTAPM database
USE astapm;

DROP TABLE data_source;

-- Create the data_source table
CREATE TABLE data_source (
    ds_id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    ds_name VARCHAR(50) NOT NULL,
    ds_description VARCHAR(255) DEFAULT NULL,
    ds_im_path VARCHAR(255) DEFAULT NULL,
    ds_ex_path VARCHAR(255) DEFAULT NULL,
    ds_type VARCHAR(50) NOT NULL,
    ds_raw_db VARCHAR(50) NOT NULL,
    ds_db VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_ds_name` (`ds_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO data_source (ds_name, ds_description, ds_im_path, ds_ex_path, ds_type, ds_raw_db, ds_db)
VALUES
('ASRS', 'Aviation Safety Reporting System', 'data/local_im/asrs', 'data/local_ex/asrs', 'csv', 'asrs_raw', 'asrs_processed'),
('NTSB', 'National Transportation Safety Board', 'data/local_im/ntsb', 'data/local_ex/ntsb',  'mdb', 'ntsb_raw', 'asrs_processed');

-- Create the model table
CREATE TABLE ml_model (
    ml_model_id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    ml_model_name VARCHAR(50) NOT NULL,
    ml_model_type ENUM('SVM', 'LSTM') NOT NULL,  -- "SVM" or "LSTM"
    ml_model_description VARCHAR(255) DEFAULT NULL,
    feature_engineering ENUM('NMF', 'ADSGL') NOT NULL, -- Default feature engineering technique
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_ml_model_name` (`ml_model_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert record for SVM model
INSERT INTO ml_model (ml_model_name, ml_model_type, ml_model_description, feature_engineering)
VALUES ('SVM_Classifier', 'SVM', 'Support Vector Machine for classification tasks', 'NMF');
INSERT INTO ml_model (ml_model_name, ml_model_type, ml_model_description, feature_engineering)
VALUES ('LSTM_Predictor', 'LSTM', 'Long Short-Term Memory network for time-series prediction', 'ADSGL');

-- Create the train_model table
CREATE TABLE train_model (
    train_model_id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    ml_model_id INT UNSIGNED,
    training_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_train_model_id` (`train_model_id`),
    FOREIGN KEY (ml_model_id) REFERENCES ml_model(ml_model_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


