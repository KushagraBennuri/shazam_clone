CREATE DATABASE IF NOT EXISTS shazam_clone
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE shazam_clone;

CREATE TABLE IF NOT EXISTS songs (
    id        INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    title     VARCHAR(255) NOT NULL,
    artist    VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64)  NOT NULL UNIQUE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS fingerprints (
    hash_val  INT UNSIGNED NOT NULL,
    song_id   INT          NOT NULL,
    offset_ms INT UNSIGNED NOT NULL,
    FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE,
    INDEX idx_hash (hash_val)
) ENGINE=InnoDB;
