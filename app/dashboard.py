from flask import Flask, render_template, send_from_directory
import sqlite3
import os

app = Flask(__name__, template_folder="templates")
DB = "violations.db"

def fetch_violations(limit=100):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, image_path, timestamp, missing_ppe, confidence, location FROM violations ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

@app.route("/")
def index():
    rows = fetch_violations()
    return render_template("dashboard.html", rows=rows)

@app.route("/violations/<path:filename>")
def violation_image(filename):
    return send_from_directory(os.path.join(app.root_path, "..", "violations"), filename)

if __name__ == "__main__":
    app.run(debug=True)
