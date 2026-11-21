import sqlite3
DB = "violations.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        missing_ppe TEXT,
        confidence REAL,
        location TEXT
      )
    """)
    conn.commit()
    conn.close()

def insert_violation(image_path, location, missing_ppe, confidence):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
      INSERT INTO violations (image_path, missing_ppe, confidence, location)
      VALUES (?, ?, ?, ?)
    """, (image_path, missing_ppe, confidence, location))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB)
