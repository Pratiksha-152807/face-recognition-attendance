
import sys
import subprocess
from pathlib import Path

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    flash,
    get_flashed_messages,
)

# Project settings
BASE = Path(__file__).resolve().parent
REGISTER_SCRIPT = BASE / "register_from_dashboard.py"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    raise RuntimeError(
        "Please set the FLASK_SECRET_KEY environment variable."
    )


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Attendance - Registration</title>

    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #eef2ff;
            color: #1e293b;
        }

        header {
            background: #172554;
            color: white;
            text-align: center;
            padding: 28px 15px;
        }

        header h1 { margin: 0 0 8px; }
        header p { margin: 0; }

        .card {
            width: min(92%, 520px);
            margin: 40px auto;
            padding: 28px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(15, 23, 42, .1);
        }

        label {
            display: block;
            margin: 18px 0 8px;
            font-weight: bold;
        }

        input {
            width: 100%;
            padding: 13px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 16px;
        }

        button {
            width: 100%;
            margin-top: 24px;
            padding: 14px;
            border: 0;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover { background: #1d4ed8; }

        .note {
            margin-top: 18px;
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
        }

        .result {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            padding: 14px;
            border-radius: 8px;
            background: #f1f5f9;
            margin-bottom: 18px;
        }

        .result.success { background: #dcfce7; }
        .result.error { background: #fee2e2; }
    </style>
</head>

<body>
    <header>
        <h1>Face Recognition Attendance</h1>
        <p>Student Registration Dashboard</p>
    </header>

    <main class="card">
        <h2>Register a Student</h2>
        <p>Enter student details to start face registration.</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% for category, message in messages %}
                <div class="result {{ category }}">{{ message }}</div>
                <script>
                    window.addEventListener("DOMContentLoaded", function () {
                        alert({{ message | tojson }});
                    });
                </script>
            {% endfor %}
        {% endwith %}

        <form action="{{ url_for('register_student') }}" method="POST">
            <label for="student_id">Student ID</label>
            <input
                id="student_id"
                name="student_id"
                type="number"
                min="1"
                required
                placeholder="Example: 110"
            >

            <label for="name">Student Name</label>
            <input
                id="name"
                name="name"
                type="text"
                maxlength="100"
                required
                placeholder="Enter full name"
            >

            <button type="submit">Start Face Registration</button>
        </form>

        <p class="note">
            The webcam will open after you submit the form.
            Keep your face visible while 50 images are captured.
            A popup will report the registration result.
        </p>
    </main>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE)


@app.route("/register", methods=["POST"])
def register_student():
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()

    if not student_id.isdigit() or int(student_id) < 1 or not name:
        flash("Please enter a valid student ID and name.", "error")
        return redirect(url_for("home"))

    if not REGISTER_SCRIPT.is_file():
        flash(
            "Cannot find register_from_dashboard.py. "
            "Make sure it is in the project folder.",
            "error",
        )
        return redirect(url_for("home"))

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(REGISTER_SCRIPT),
                student_id,
                name,
            ],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
            check=False,
        )

        output = (result.stdout or "").strip()
        errors = (result.stderr or "").strip()
        combined = output
        if errors:
            combined += ("\n" if combined else "") + errors

        # Determine the result from the registration script's output.
        if "DUPLICATE FACE DETECTED!" in combined:
            flash(
                "DUPLICATE FACE DETECTED!\n\n"
                + combined
                + "\n\nRegistration was cancelled.",
                "error",
            )

        elif (
            result.returncode == 0
            and "REGISTRATION SUCCESSFUL!" in combined
        ):
            flash(
                "REGISTRATION SUCCESSFUL!\n\n" + combined,
                "success",
            )

        elif result.returncode != 0:
            flash(
                "REGISTRATION CANCELLED OR FAILED.\n\n"
                + (combined or "The script returned no details."),
                "error",
            )

        else:
            flash(
                "Registration result could not be confirmed.\n\n"
                + (combined or "No output received."),
                "error",
            )

    except subprocess.TimeoutExpired:
        flash(
            "Registration timed out. Check your webcam and student "
            "records before trying again.",
            "error",
        )

    except Exception as error:
        flash("Could not run registration: " + str(error), "error")

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
