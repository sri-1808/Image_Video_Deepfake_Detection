import streamlit as st
import subprocess

def run_app(app_file, port):
    subprocess.Popen(["streamlit", "run", app_file, "--server.port", str(port)])

st.title('Choose an Application to Run')

# app_launcher.py
if st.button('Run Video Application'):
    st.write('Running video application...')
    run_app('app_video.py', 8503)  # Use the new port for video app

if st.button('Run Image Application'):
    st.write('Running image application...')
    run_app('app_image.py', 8504)  # Use the new port for image app
