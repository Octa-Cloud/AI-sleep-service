-- Analysis Service Database Schema
-- Database: mong-analysis

USE `mong-analysis`;

-- Create ENUM types first
-- Note: MySQL doesn't support CREATE TYPE, so we'll use ENUM directly in table definitions

-- 1. Sleep Sessions Table
CREATE TABLE IF NOT EXISTS `sleep_sessions` (
    `sleep_session_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `user_no` BIGINT NOT NULL,
    `finished_at` DATETIME NULL,
    `created_at` DATETIME NOT NULL,
    INDEX `idx_sleep_sessions_user_no` (`user_no`),
    INDEX `idx_sleep_sessions_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Daily Reports Table
CREATE TABLE IF NOT EXISTS `daily_reports` (
    `sleep_session_no` BIGINT PRIMARY KEY,
    `memo` VARCHAR(255) NULL,
    `user_no` BIGINT NOT NULL,
    `created_at` DATETIME NOT NULL,
    FOREIGN KEY (`sleep_session_no`) REFERENCES `sleep_sessions`(`sleep_session_no`) ON DELETE CASCADE,
    INDEX `idx_daily_reports_user_no` (`user_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Sleep Time Details Table
CREATE TABLE IF NOT EXISTS `sleep_time_details` (
    `sleep_session_no` BIGINT PRIMARY KEY,
    `deep_sleep_minutes` SMALLINT UNSIGNED NOT NULL,
    `light_sleep_minutes` SMALLINT UNSIGNED NOT NULL,
    `rem_sleep_minutes` SMALLINT UNSIGNED NOT NULL,
    `deep_sleep_ratio` DECIMAL(4,1) NOT NULL,
    `light_sleep_ratio` DECIMAL(4,1) NOT NULL,
    `rem_sleep_ratio` DECIMAL(4,1) NOT NULL,
    FOREIGN KEY (`sleep_session_no`) REFERENCES `daily_reports`(`sleep_session_no`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Analysis Details Table
CREATE TABLE IF NOT EXISTS `analysis_details` (
    `analysis_detail_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `sleep_session_no` BIGINT NOT NULL,
    `title` VARCHAR(255) NOT NULL,
    `description` VARCHAR(255) NOT NULL,
    `difficulty` ENUM('EASY', 'MEDIUM', 'HARD') NOT NULL,
    `effect` ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    FOREIGN KEY (`sleep_session_no`) REFERENCES `daily_reports`(`sleep_session_no`) ON DELETE CASCADE,
    INDEX `idx_analysis_details_sleep_session_no` (`sleep_session_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Analysis Steps Table
CREATE TABLE IF NOT EXISTS `analysis_steps` (
    `analysis_step_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `analysis_detail_no` BIGINT NOT NULL,
    `step_index` SMALLINT UNSIGNED NOT NULL,
    `content` VARCHAR(255) NOT NULL,
    FOREIGN KEY (`analysis_detail_no`) REFERENCES `analysis_details`(`analysis_detail_no`) ON DELETE CASCADE,
    INDEX `idx_analysis_steps_analysis_detail_no` (`analysis_detail_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. Periodic Reports Table
CREATE TABLE IF NOT EXISTS `periodic_reports` (
    `periodic_report_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `user_no` BIGINT NOT NULL,
    `sleep_session_count` SMALLINT UNSIGNED NOT NULL,
    `duration_type` ENUM('WEEKLY', 'MONTHLY') NOT NULL,
    `period_started_at` DATE NOT NULL,
    `total_score` SMALLINT UNSIGNED NOT NULL,
    `total_sleep_time` SMALLINT UNSIGNED NOT NULL,
    `total_bed_time_minutes` SMALLINT UNSIGNED NOT NULL,
    `total_deep_sleep_time_minutes` SMALLINT UNSIGNED NOT NULL,
    `total_light_sleep_time_minutes` SMALLINT UNSIGNED NOT NULL,
    `total_rem_sleep_time_minutes` SMALLINT UNSIGNED NOT NULL,
    `improvement` VARCHAR(500) NULL,
    `weakness` VARCHAR(500) NULL,
    `recommendation` VARCHAR(500) NULL,
    `score_prediction_description` VARCHAR(500) NULL,
    INDEX `idx_periodic_reports_user_no` (`user_no`),
    INDEX `idx_periodic_reports_period_started_at` (`period_started_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. Score Prediction Points Table
CREATE TABLE IF NOT EXISTS `score_prediction_points` (
    `score_prediction_point_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `periodic_report_no` BIGINT NOT NULL,
    `date_index` DATE NOT NULL,
    `score` SMALLINT NOT NULL,
    FOREIGN KEY (`periodic_report_no`) REFERENCES `periodic_reports`(`periodic_report_no`) ON DELETE CASCADE,
    INDEX `idx_score_prediction_points_periodic_report_no` (`periodic_report_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. Analyzed Sound Events Table
CREATE TABLE IF NOT EXISTS `analyzed_sound_events` (
    `analyzed_sound_event_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `sleep_session_no` BIGINT NULL,
    `event` ENUM('SNORE', 'BABY_CRYING', 'COUGH', 'MOUTH_BREATHING', 'ANIMAL_NOISE', 'CAR_HORN') NULL,
    `recorded_at` DATETIME NOT NULL,
    FOREIGN KEY (`sleep_session_no`) REFERENCES `sleep_sessions`(`sleep_session_no`),
    INDEX `idx_analyzed_sound_events_sleep_session_no` (`sleep_session_no`),
    INDEX `idx_analyzed_sound_events_recorded_at` (`recorded_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. Analyzed Sleep Levels Table
CREATE TABLE IF NOT EXISTS `analyzed_sleep_levels` (
    `analyzed_sleep_level_no` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `sleep_session_no` BIGINT NULL,
    `level` SMALLINT UNSIGNED NULL,
    `recorded_at` DATETIME NOT NULL,
    FOREIGN KEY (`sleep_session_no`) REFERENCES `sleep_sessions`(`sleep_session_no`),
    CONSTRAINT `chk_analyzed_sleep_levels_level` CHECK (`level` >= 0 AND `level` <= 6),
    UNIQUE KEY `uq_analyzed_sleep_levels_session_time` (`sleep_session_no`, `recorded_at`),
    INDEX `idx_analyzed_sleep_levels_sleep_session_no` (`sleep_session_no`),
    INDEX `idx_analyzed_sleep_levels_recorded_at` (`recorded_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Show created tables
SHOW TABLES;
