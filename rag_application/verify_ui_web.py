import os
import subprocess
import time


def run_streamlit_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, '..', 'rag_application/modules/user_interface.py')

    # Command to run the Streamlit app
    command = ["streamlit", "run", app_path]

    # Launch the Streamlit app
    process = subprocess.Popen(command)

    # Give it some time to start
    time.sleep(10)

    # Check if the process is still running
    if process.poll() is None:
        print("Streamlit app is running")
    else:
        print("Failed to start the Streamlit app")

if __name__ == "__main__":
    run_streamlit_app()
