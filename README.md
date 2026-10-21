\# Face Recognition Attendance System



\## Project Overview



The Face Recognition Attendance System is a Python-based application that uses computer vision to recognize registered students and record their attendance. It includes a web dashboard for managing students and viewing attendance records.



This project was developed as an academic project to demonstrate the use of Artificial Intelligence, OpenCV, Python, and Flask.



\## Features



\- Student face registration

\- Face recognition using OpenCV

\- Automatic attendance recording

\- Flask-based web dashboard

\- Student management

\- Date-wise attendance filtering

\- Student-wise attendance history

\- Attendance reports and analytics

\- CSV attendance export



\## Technologies Used



\- Python

\- OpenCV

\- Flask

\- NumPy

\- HTML and CSS

\- JavaScript and Chart.js

\- CSV for data storage



\## Project Requirements



\- Python 3.11 or a compatible Python version

\- A webcam

\- Required Python packages listed in `requirements.txt`

\- A computer capable of running OpenCV



\## Installation and Setup



\### 1. Clone the repository



Replace `YOUR-USERNAME` with your GitHub username.



```bash

git clone https://github.com/YOUR-USERNAME/face-recognition-attendance.git

cd face-recognition-attendance

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



Activate it on Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

python -m pip install -r requirements.txt

```



\### 4. Prepare the project



\- Ensure the Haar Cascade XML file is present.

\- Register students using the project's registration workflow.

\- Capture face images for the registered students.

\- Train the recognition model before running attendance recognition.



The face dataset, trained model, student records, and attendance records are not included in this repository. Generate them locally using the application's workflow.



\### 5. Run the dashboard



```bash

python dashboard.py

```



Open the following address in your browser:



http://127.0.0.1:5000



Use the application's configured login credentials. Change any default credentials before deployment.



\## Privacy and Security



Face images are biometric data. Do not upload real student photographs, face datasets, student details, attendance records, passwords, secret keys, or private backups to GitHub.



Use fictional student data for demonstrations. Obtain appropriate permission before collecting or processing real students' biometric information.



\## Limitations



\- Recognition accuracy can be affected by lighting, camera quality, face angle, and image quality.

\- A webcam is required for live face registration and recognition.

\- Attendance records depend on the application's local data files.

\- This academic prototype should be reviewed and secured before real-world deployment.



\## Academic Purpose



This project demonstrates the practical application of computer vision, face recognition, web development, and attendance management.





