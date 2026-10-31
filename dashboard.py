
import csv
import os
import sys
import subprocess
from pathlib import Path
from datetime import date, datetime
from functools import wraps

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    session,
    flash,
    render_template_string,
    send_file,
)

# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    raise RuntimeError(
        "Please set the FLASK_SECRET_KEY environment variable."
    )
BASE_DIR = Path(__file__).resolve().parent

DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

if DEMO_MODE:
    STUDENTS_FILE = BASE_DIR / "demo_data" / "students.csv"
    ATTENDANCE_FILE = BASE_DIR / "demo_data" / "attendance.csv"
else:
    STUDENTS_FILE = BASE_DIR / "students.csv"
    ATTENDANCE_FILE = BASE_DIR / "attendance.csv"


TRAIN_SCRIPT = BASE_DIR / "train_model.py"
ATTENDANCE_SCRIPT = BASE_DIR / "attendance.py"
REGISTER_SCRIPT = BASE_DIR / "register_from_dashboard.py"

MODEL_FILE = BASE_DIR / "models" / "face_model.yml"

# Set these environment variables to customize login credentials.
ADMIN_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")

if not ADMIN_PASSWORD:
    raise RuntimeError(
        "Please set the DASHBOARD_PASSWORD environment variable."
    )


# ==========================================================
# 2. CSV HELPERS
# ==========================================================

def normalize(value):
    """Normalize CSV headers and values for comparison."""
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def read_csv_file(file_path):
    """Read a CSV file safely and return (headers, rows)."""
    if not file_path.exists():
        return [], []

    try:
        with file_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            rows = [
                {str(k): (v or "").strip() for k, v in row.items() if k is not None}
                for row in reader
            ]
            return headers, rows
    except (OSError, UnicodeError, csv.Error):
        return [], []


def find_value(row, possible_names, default=""):
    """Find a value even if the CSV uses a slightly different header."""
    normalized_row = {normalize(k): v for k, v in row.items()}

    for name in possible_names:
        key = normalize(name)
        if key in normalized_row and normalized_row[key] != "":
            return normalized_row[key]

    return default


def student_id_of(row):
    return find_value(
        row,
        ["student_id", "studentid", "id", "roll_no", "roll_number", "roll"],
    )


def student_name_of(row):
    return find_value(
        row,
        ["name", "student_name", "studentname", "full_name"],
        "Unknown",
    )


def attendance_date_of(row):
    return find_value(
        row,
        ["date", "attendance_date", "day"],
    )


def attendance_time_of(row):
    return find_value(
        row,
        ["time", "attendance_time", "timestamp", "datetime"],
    )


def parse_date(value):
    """Parse common date formats without crashing on invalid values."""
    value = str(value or "").strip()

    if not value:
        return None

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def get_students():
    _, rows = read_csv_file(STUDENTS_FILE)

    students = []
    seen = set()

    for row in rows:
        sid = student_id_of(row)
        name = student_name_of(row)

        if not sid:
            continue

        if sid not in seen:
            students.append({
                "id": sid,
                "name": name,
                "email": find_value(row, ["email", "email_address"]),
            })
            seen.add(sid)

    students.sort(key=lambda student: student["name"].lower())
    return students


def get_attendance():
    _, rows = read_csv_file(ATTENDANCE_FILE)

    records = []

    for row in rows:
        sid = student_id_of(row)
        raw_date = attendance_date_of(row)
        raw_time = attendance_time_of(row)
        parsed = parse_date(raw_date)

        # Some files store the full date and time in one column.
        if parsed is None:
            parsed = parse_date(raw_time)

        if not sid or parsed is None:
            continue

        records.append({
            "id": sid,
            "name": find_value(
                row,
                ["name", "student_name", "studentname", "full_name"],
                "Unknown",
            ),
            "date": parsed.isoformat(),
            "time": raw_time,
            "status": find_value(row, ["status"], "Present") or "Present",
        })

    return records


def attendance_percentage(student_id, records):
    """Percentage of calendar days with attendance in the available data."""
    student_dates = {
        record["date"]
        for record in records
        if record["id"] == student_id
    }

    if not student_dates:
        return 0

    first_day = min(parse_date(d) for d in student_dates)
    last_day = max(parse_date(d) for d in student_dates)
    total_days = (last_day - first_day).days + 1

    if total_days <= 0:
        return 0

    return round(len(student_dates) / total_days * 100, 1)


# ==========================================================
# 3. LOGIN PROTECTION
# ==========================================================

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


# ==========================================================
# 4. SHARED HTML DESIGN
# ==========================================================

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }} | Attendance System</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6fb;
            color: #202b3c;
        }
        header {
            background: #172554;
            color: white;
            padding: 20px 5%;
        }
        header h1 { margin: 0 0 7px; font-size: 24px; }
        header p { margin: 0; color: #dbeafe; }
        nav {
            background: white;
            padding: 14px 5%;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            border-bottom: 1px solid #e5e7eb;
        }
        nav a {
            text-decoration: none;
            color: #1d4ed8;
            font-weight: bold;
        }
        main { max-width: 1250px; margin: 25px auto; padding: 0 18px; }
        h2 { margin-top: 0; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin-bottom: 22px;
        }
        .card, .panel {
            background: white;
            padding: 22px;
            border-radius: 14px;
            box-shadow: 0 3px 14px rgba(15, 23, 42, .06);
            margin-bottom: 20px;
        }
        .card p { margin: 0 0 10px; color: #64748b; }
        .card strong { font-size: 30px; color: #1d4ed8; }
        .muted { color: #64748b; font-size: 13px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        input, select {
            padding: 11px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin: 5px 4px 5px 0;
            max-width: 100%;
        }
        button, .button {
            display: inline-block;
            background: #2563eb;
            color: white;
            border: 0;
            border-radius: 8px;
            padding: 11px 15px;
            text-decoration: none;
            cursor: pointer;
            margin: 4px 3px 4px 0;
            font-size: 14px;
        }
        .secondary { background: #475569; }
        .danger { background: #b91c1c; }
        .success { background: #047857; }
        .table-wrap { overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; background: white; }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
            white-space: nowrap;
        }
        th { background: #eff6ff; }
        .flash {
            padding: 12px 15px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .login {
            max-width: 420px;
            margin: 70px auto;
            padding: 28px;
            background: white;
            border-radius: 14px;
        }
        .login input { width: 100%; margin: 8px 0; }
        .login button { width: 100%; margin-top: 12px; }
        canvas { max-height: 340px; }
        footer { text-align: center; color: #64748b; padding: 25px; }
        @media (max-width: 600px) {
            header h1 { font-size: 20px; }
            .card, .panel { padding: 16px; }
        }
    </style>
</head>
<body>
    <header>
        <h1>Face Recognition Attendance</h1>
        <p>Smart Student Attendance Management System</p>
    </header>

    {% if session.get('logged_in') %}
    <nav>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('students_page') }}">Students</a>
        <a href="{{ url_for('reports') }}">Reports</a>
        <a href="{{ url_for('analytics') }}">Analytics</a>
        <a href="{{ url_for('attendance_percentages') }}">Attendance %</a>
        <a href="{{ url_for('logout') }}">Logout</a>
    </nav>
    {% endif %}

    <main>
        {% with messages = get_flashed_messages() %}
            {% for message in messages %}
                <div class="flash">{{ message }}</div>
            {% endfor %}
        {% endwith %}
        {{ content|safe }}
    </main>
    <footer>Face Recognition Attendance Management System</footer>
</body>
</html>
"""


def render_page(title, content, **context):
    inner = render_template_string(content, **context)
    return render_template_string(
        BASE_HTML, title=title, content=inner
    )


# ==========================================================
# 5. LOGIN AND LOGOUT
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["logged_in"] = True
            return redirect(url_for("dashboard"))

        flash("Incorrect username or password.")

    return render_page("Login", """
        <div class="login">
            <h2>Admin Login</h2>
            <p class="muted">Sign in to manage attendance records.</p>
            <form method="post">
                <label>Username</label>
                <input name="username" required autocomplete="username">
                <label>Password</label>
                <input type="password" name="password" required
                       autocomplete="current-password">
                <button type="submit">Login</button>
            </form>
        </div>
    """)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==========================================================
# 6. DASHBOARD
# ==========================================================

@app.route("/")
@login_required
def dashboard():
    students = get_students()
    records = get_attendance()
    today = date.today().isoformat()

    present_today = {
        record["id"]
        for record in records
        if record["date"] == today
    }

    daily_counts = {}
    for record in records:
        daily_counts.setdefault(record["date"], set()).add(record["id"])

    labels = sorted(daily_counts.keys())[-14:]
    values = [len(daily_counts[day]) for day in labels]

    recent = sorted(
        records,
        key=lambda record: (record["date"], record["time"]),
        reverse=True,
    )[:10]

    return render_page("Dashboard", """
        <h2>Dashboard</h2>
        <p class="muted">Today: {{ today }}</p>

        <div class="cards">
            <div class="card">
                <p>Total Registered Students</p>
                <strong>{{ students|length }}</strong>
            </div>
            <div class="card">
                <p>Students Present Today</p>
                <strong>{{ present }}</strong>
            </div>
            <div class="card">
                <p>Total Attendance Records</p>
                <strong>{{ records|length }}</strong>
            </div>
        </div>

        <div class="panel">
    <h2>Quick Actions</h2>
    <a class="button" href="{{ url_for('students_page') }}">Manage Students</a>
    <a class="button secondary" href="{{ url_for('reports') }}">View Reports</a>
    <a class="button secondary" href="{{ url_for('analytics') }}">View Analytics</a>
    <a class="button success" href="{{ url_for('export_csv') }}">Export CSV</a>
    <hr>

    {% if not demo_mode %}
    <form method="post" action="{{ url_for('run_script', script_name='train') }}">
        <button type="submit">Train Face Model</button>
    </form>

    <form method="post" action="{{ url_for('run_script', script_name='attendance') }}">
        <button type="submit" class="success">Start Face Recognition</button>
    </form>

    <p class="muted">
        Face recognition opens a camera window when supported.
        Close that window using the key expected by your attendance.py.
    </p>
    {% else %}
    <p class="muted">
        This is an online demonstration using fictional student records.
        Face recognition is available in the local application only.
    </p>
    {% endif %}
</div>
        <div class="grid">
            <div class="panel">
                <h2>Daily Attendance</h2>
                <canvas id="dailyChart"></canvas>
            </div>
            <div class="panel">
                <h2>Recent Attendance</h2>
                <div class="table-wrap">
                    <table>
                        <tr><th>Date</th><th>Student ID</th><th>Name</th><th>Time</th></tr>
                        {% for r in recent %}
                        <tr>
                            <td>{{ r.date }}</td><td>{{ r.id }}</td>
                            <td>{{ r.name }}</td><td>{{ r.time }}</td>
                        </tr>
                        {% else %}
                        <tr><td colspan="4">No valid attendance records found.</td></tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
        </div>

        <script>
        new Chart(document.getElementById('dailyChart'), {
            type: 'line',
            data: {
                labels: {{ labels|tojson }},
                datasets: [{
                    label: 'Students present',
                    data: {{ values|tojson }},
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37,99,235,0.15)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
        </script>
    """, students=students, records=records, today=today,
         present=len(present_today), recent=recent, labels=labels, values=values, demo_mode=DEMO_MODE)


# ==========================================================
# 7. STUDENT LIST AND SEARCH
# ==========================================================

@app.route("/students")
@login_required
def students_page():
    query = request.args.get("search", "").strip().lower()
    students = get_students()

    if query:
        students = [
            student for student in students
            if query in student["id"].lower()
            or query in student["name"].lower()
        ]

    return render_page("Students", """
        <h2>Registered Students</h2>
        <div class="panel">
            <form method="get">
                <input name="search" placeholder="Search name or student ID"
                       value="{{ query }}">
                <button type="submit">Search</button>
                <a class="button secondary" href="{{ url_for('students_page') }}">Clear</a>
            </form>
            <div class="table-wrap">
                <table>
                    <tr><th>Student ID</th><th>Name</th><th>Email</th><th>Action</th></tr>
                    {% for s in students %}
                    <tr>
                        <td>{{ s.id }}</td>
                        <td>{{ s.name }}</td>
                        <td>{{ s.email or '-' }}</td>
                        <td><a href="{{ url_for('student_history', student_id=s.id) }}">View History</a></td>
                    </tr>
                    {% else %}
                    <tr><td colspan="4">No students found.</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <div class="panel">
            <h2>Register Student</h2>
            <p class="muted">
                This form saves student details to students.csv. It does not
                capture face images. Use your face-registration script to
                capture images and update the face dataset before training.
            </p>
            <form method="post" action="{{ url_for('add_student') }}">
                <input name="student_id" placeholder="Student ID" required>
                <input name="name" placeholder="Student name" required>
                <input name="email" type="email" placeholder="Email (optional)">
                <button type="submit">Save Student Details</button>
            </form>
        </div>
    """, students=students, query=query)



@app.route("/students/add", methods=["POST"])
@login_required
def add_student():
    sid = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()

    if not sid.isdigit() or not name:
        flash("Enter a numeric student ID and a valid name.")
        return redirect(url_for("students_page"))

    existing = get_students()

    if any(student["id"] == sid for student in existing):
        flash("That student ID already exists.")
        return redirect(url_for("students_page"))

    # Online demo: save student details without using a webcam.
    if DEMO_MODE:
        headers, rows = read_csv_file(STUDENTS_FILE)

        if not headers:
            headers = ["student_id", "name"]

        if email and "email" not in headers:
            headers.append("email")

        new_row = {
            "student_id": sid,
            "name": name,
        }

        if "email" in headers:
            new_row["email"] = email

        try:
            with STUDENTS_FILE.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=headers,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(new_row)

            flash("Demo student details saved successfully.")

        except OSError as exc:
            flash(f"Could not save student: {exc}")

        return redirect(url_for("students_page"))

    # Local application: retain webcam registration.
    if not REGISTER_SCRIPT.is_file():
        flash("Face registration script was not found.")
        return redirect(url_for("students_page"))

    try:
        subprocess.Popen(
            [sys.executable, str(REGISTER_SCRIPT), sid, name],
            cwd=str(BASE_DIR),
        )
        flash(
            f"Webcam registration started for {name} ({sid}). "
            "Complete the face capture."
        )

    except OSError as exc:
        flash(f"Could not start face registration: {exc}")

    return redirect(url_for("students_page"))


# ==========================================================
# 8. STUDENT ATTENDANCE HISTORY
# ==========================================================

@app.route("/student/<student_id>")
@login_required
def student_history(student_id):
    students = get_students()
    student = next((s for s in students if s["id"] == student_id), None)

    if student is None:
        flash("Student not found.")
        return redirect(url_for("students_page"))

    records = [
        record for record in get_attendance()
        if record["id"] == student_id
    ]
    records.sort(key=lambda record: (record["date"], record["time"]), reverse=True)

    return render_page("Student History", """
        <h2>Attendance History</h2>
        <div class="panel">
            <h3>{{ student.name }}</h3>
            <p>Student ID: {{ student.id }}</p>
            <p>Total recorded attendance days: {{ days }}</p>
            <p>Attendance percentage across the recorded date span: {{ percent }}%</p>
        </div>
        <div class="panel">
            <div class="table-wrap">
                <table>
                    <tr><th>Date</th><th>Time</th><th>Status</th></tr>
                    {% for r in records %}
                    <tr><td>{{ r.date }}</td><td>{{ r.time or '-' }}</td><td>{{ r.status }}</td></tr>
                    {% else %}
                    <tr><td colspan="3">No attendance history for this student.</td></tr>
                    {% endfor %}
                </table>
            </div>
            <p><a href="{{ url_for('students_page') }}">Back to Students</a></p>
        </div>
    """, student=student, records=records,
         days=len({r["date"] for r in records}),
         percent=attendance_percentage(student_id, records))


# ==========================================================
# 9. DATE-FILTERED REPORTS
# ==========================================================

@app.route("/reports")
@login_required
def reports():
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    query = request.args.get("search", "").strip().lower()

    records = get_attendance()

    if start_date:
        if not parse_date(start_date):
            flash("Invalid start date.")
            start_date = ""
        else:
            records = [r for r in records if r["date"] >= start_date]

    if end_date:
        if not parse_date(end_date):
            flash("Invalid end date.")
            end_date = ""
        else:
            records = [r for r in records if r["date"] <= end_date]

    if query:
        records = [
            r for r in records
            if query in r["id"].lower() or query in r["name"].lower()
        ]

    records.sort(key=lambda r: (r["date"], r["time"]), reverse=True)

    return render_page("Reports", """
        <h2>Attendance Reports</h2>
        <div class="panel">
            <form method="get">
                <label>From</label>
                <input type="date" name="start_date" value="{{ start_date }}">
                <label>To</label>
                <input type="date" name="end_date" value="{{ end_date }}">
                <input name="search" placeholder="Student ID or name" value="{{ query }}">
                <button type="submit">Apply Filters</button>
                <a class="button secondary" href="{{ url_for('reports') }}">Reset</a>
            </form>
        </div>
        <div class="panel">
            <p>Total matching records: <strong>{{ records|length }}</strong></p>
            <div class="table-wrap">
                <table>
                    <tr><th>Date</th><th>Student ID</th><th>Name</th><th>Time</th><th>Status</th></tr>
                    {% for r in records %}
                    <tr>
                        <td>{{ r.date }}</td><td>{{ r.id }}</td><td>{{ r.name }}</td>
                        <td>{{ r.time or '-' }}</td><td>{{ r.status }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="5">No matching attendance records.</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    """, records=records, start_date=start_date, end_date=end_date, query=query)


# ==========================================================
# 10. ANALYTICS
# ==========================================================

@app.route("/analytics")
@login_required
def analytics():
    students = get_students()
    records = get_attendance()

    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()

    filtered = records

    if start_date and parse_date(start_date):
        filtered = [r for r in filtered if r["date"] >= start_date]

    if end_date and parse_date(end_date):
        filtered = [r for r in filtered if r["date"] <= end_date]

    daily = {}
    for record in filtered:
        daily.setdefault(record["date"], set()).add(record["id"])

    daily_labels = sorted(daily.keys())
    daily_values = [len(daily[d]) for d in daily_labels]

    student_labels = [student["name"] for student in students]
    student_values = [
        round(
            len({r["date"] for r in filtered if r["id"] == student["id"]})
            / max(len({r["date"] for r in filtered}), 1) * 100,
            1,
        )
        for student in students
    ]

    return render_page("Analytics", """
        <h2>Attendance Analytics</h2>
        <div class="panel">
            <form method="get">
                <label>From</label>
                <input type="date" name="start_date" value="{{ start_date }}">
                <label>To</label>
                <input type="date" name="end_date" value="{{ end_date }}">
                <button type="submit">Update Charts</button>
                <a class="button secondary" href="{{ url_for('analytics') }}">Reset</a>
            </form>
            <p class="muted">
                Percentages compare each student's distinct attendance dates
                with the distinct dates in the selected report period. They
                are not official college attendance percentages unless the
                period contains only actual working days.
            </p>
        </div>

        <div class="grid">
            <div class="panel">
                <h3>Daily Attendance</h3>
                <canvas id="dailyChart"></canvas>
            </div>
            <div class="panel">
                <h3>Student Attendance Percentage</h3>
                <canvas id="studentChart"></canvas>
            </div>
        </div>

        <script>
        new Chart(document.getElementById('dailyChart'), {
            type: 'bar',
            data: {
                labels: {{ daily_labels|tojson }},
                datasets: [{
                    label: 'Distinct students present',
                    data: {{ daily_values|tojson }},
                    backgroundColor: '#3b82f6'
                }]
            },
            options: { responsive: true }
        });

        new Chart(document.getElementById('studentChart'), {
            type: 'bar',
            data: {
                labels: {{ student_labels|tojson }},
                datasets: [{
                    label: 'Attendance percentage',
                    data: {{ student_values|tojson }},
                    backgroundColor: '#10b981'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                scales: {
                    x: { min: 0, max: 100 }
                }
            }
        });
        </script>
    """, start_date=start_date, end_date=end_date,
         daily_labels=daily_labels, daily_values=daily_values,
         student_labels=student_labels, student_values=student_values)


# ==========================================================
# 11. ATTENDANCE PERCENTAGE TABLE
# ==========================================================

@app.route("/attendance-percentages")
@login_required
def attendance_percentages():
    students = get_students()
    records = get_attendance()

    table_rows = []

    for student in students:
        student_records = [
            r for r in records if r["id"] == student["id"]
        ]
        distinct_days = len({r["date"] for r in student_records})

        table_rows.append({
            "id": student["id"],
            "name": student["name"],
            "days": distinct_days,
            "percentage": attendance_percentage(student["id"], records),
        })

    return render_page("Attendance Percentages", """
        <h2>Attendance Percentages</h2>
        <div class="panel">
            <p class="muted">
                This estimate uses the calendar span between each student's
                first and last recorded attendance. It does not know the
                college's actual working-day calendar.
            </p>
            <div class="table-wrap">
                <table>
                    <tr><th>Student ID</th><th>Name</th><th>Recorded Days</th><th>Estimated Percentage</th></tr>
                    {% for r in rows %}
                    <tr>
                        <td>{{ r.id }}</td><td>{{ r.name }}</td>
                        <td>{{ r.days }}</td><td>{{ r.percentage }}%</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="4">No students registered.</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    """, rows=table_rows)


# ==========================================================
# 12. CSV EXPORT
# ==========================================================

@app.route("/export")
@login_required
def export_csv():
    if not ATTENDANCE_FILE.exists():
        flash("attendance.csv does not exist yet.")
        return redirect(url_for("reports"))

    return send_file(
        ATTENDANCE_FILE,
        as_attachment=True,
        download_name="attendance_records.csv",
        mimetype="text/csv",
    )


# ==========================================================
# 13. RUN EXISTING PROJECT SCRIPTS
# ==========================================================


@app.route("/run/<script_name>", methods=["POST"])
@login_required
def run_script(script_name):
    if DEMO_MODE:
        flash("Face recognition is available only in the local application.")
        return redirect(url_for("dashboard"))
    scripts = {
        "train": TRAIN_SCRIPT,
        "attendance": ATTENDANCE_SCRIPT,
    }

    script_path = scripts.get(script_name)

    if script_path is None:
        flash(
            "Use the student registration form to register a face."
            if script_name == "register"
            else "Unknown script."
        )
        return redirect(url_for("dashboard"))

    if not script_path.is_file():
        flash(f"Script not found: {script_path.name}")
        return redirect(url_for("dashboard"))

    if script_name == "train":
        students = get_students()
        if len(students) < 2:
            flash("Register at least two students before training.")
            return redirect(url_for("dashboard"))

    try:
        subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
        )
        flash(
            f"{script_path.name} started. "
            "Check the terminal or camera window."
        )
    except OSError as exc:
        flash(f"Could not start {script_path.name}: {exc}")

    return redirect(url_for("dashboard"))


    try:
        # Runs the script with the same Python interpreter as this app.
        # The working directory is the project directory.
        subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
        )
        flash(
            f"{script_path.name} started. Check the terminal or camera window "
            "for its output."
        )
    except OSError as exc:
        flash(f"Could not start {script_path.name}: {exc}")

    return redirect(url_for("dashboard"))


# ==========================================================
# 14. ERROR HANDLERS
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):
    return render_page("Not Found", """
        <div class="panel">
            <h2>Page not found</h2>
            <p>The page you requested does not exist.</p>
            <a class="button" href="{{ url_for('dashboard') }}">Back to Dashboard</a>
        </div>
    """), 404


# ==========================================================
# 15. START APPLICATION
# ==========================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Face Recognition Attendance Dashboard")
    print(f"Project folder: {BASE_DIR}")
    print(f"Students file: {STUDENTS_FILE}")
    print(f"Attendance file: {ATTENDANCE_FILE}")
    print("Open http://127.0.0.1:5000 in your browser.")
    
    print("=" * 55)

    app.run(host="127.0.0.1", port=5000, debug=True)
