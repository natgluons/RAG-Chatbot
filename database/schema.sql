CREATE TABLE IF NOT EXISTS user_interactions (
    user_id INTEGER PRIMARY KEY,
    request_count INTEGER,
    last_request_time TEXT
);
