import os
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collect_images import main as capture_images
from create_dataset import create_dataset
from train_classifier import train
from inference_classifier import inference
from rpi_motor_control import control_motor
from tkinter import Toplevel, Canvas
from inference_classifier import inference
import threading
import logging
import cv2
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Global variables
root = tk.Tk()
root.title("Dataset Generator & Conveyor Control")
root.geometry("700x900")
root.configure(bg="#2e3b4e")

data_dir = Path('./data')
cam_id = 0
class_names = []

task_var = tk.IntVar()
task_var.set(1)

model_file = None
motor_thread = None
stop_motor = threading.Event()
is_auto_running = False

button_style = {
    "font": ("Helvetica", 12, "bold"),
    "bg": "#4a90e2",
    "fg": "#ffffff",
    "activebackground": "#357ABD",
    "activeforeground": "#ffffff",
    "relief": tk.RAISED,
    "bd": 3,
    "width": 20
}

label_style = {
    "font": ("Helvetica", 14),
    "bg": "#2e3b4e",
    "fg": "#ffffff"
}


# Helper: Safely open and release camera
def safe_open_camera(cam_id):
    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        logging.error(f"Cannot open camera {cam_id}")
        cam.release()
        raise RuntimeError(f"Cannot open camera {cam_id}")
    return cam


def test_camera():
    try:
        cam = safe_open_camera(cam_id)
        logging.info(f"Testing camera {cam_id}...")
        for _ in range(5):
            ret, frame = cam.read()
            if ret:
                logging.info("Frame captured successfully.")
            else:
                logging.error("Failed to capture frame.")
            time.sleep(1)
    finally:
        cam.release()
        logging.info(f"Camera {cam_id} released.")


# GUI Button Callbacks
def add_class_entry():
    entry = class_entry.get().strip()
    if entry and entry not in class_names:
        class_names.append(entry)
        class_entry.delete(0, tk.END)
        button_collecting_data.configure(state="normal")
    else:
        logging.warning("Class name is empty or already exists.")


def run_collecting_data():
    button_create_dataset.configure(state="disabled")

    def target():
        try:
            capture_images(cam_id=cam_id, class_names=class_names, data_dir=data_dir)
            button_create_dataset.configure(state="normal")
        except Exception as e:
            logging.error(f"Error in collecting data: {e}")
            button_collecting_data.configure(state="normal")  # Allow retry

    threading.Thread(target=target, daemon=True).start()


def run_create_dataset():
    button_train_classifier.configure(state="disabled")

    try:
        create_dataset(data_dir)
        button_train_classifier.configure(state="normal")
    except Exception as e:
        logging.error(f"Error in creating dataset: {e}")



def run_train_classifier():
    button_inference_classifier.configure(state="disabled")
    button_safe_model.configure(state="disabled")

    global model_file
    try:
        model_file = train(data_dir)
        button_inference_classifier.configure(state="normal")
        button_safe_model.configure(state="normal")
    except Exception as e:
        logging.error(f"Error in training model: {e}")


def run_inference_classifier():
    
    def target():
        try:
            # Neues Fenster für die Live-Ansicht erstellen
            inference_window = Toplevel(root)
            inference_window.title("Live View")
            canvas = Canvas(inference_window, width=320, height=240)
            canvas.pack()

            # Live-Ansicht starten
            inference(data_dir, canvas, inference_window)
        except Exception as e:
            logging.error(f"Error in inference: {e}")
    
    threading.Thread(target=target, daemon=True).start()


def safe_model_file():
    dateiname = "model.pickle"
    ursprungspfad = data_dir / dateiname

    if not ursprungspfad.exists():
        logging.error(f"Model file '{ursprungspfad}' does not exist.")
        return

    zielpfad = filedialog.askdirectory(title="Choose target directory")
    if zielpfad:
        zielpfad = Path(zielpfad) / dateiname
        try:
            shutil.copy(ursprungspfad, zielpfad)
            logging.info(f"File copied from '{ursprungspfad}' to '{zielpfad}'.")
        except Exception as e:
            logging.error(f"Error copying file: {e}")


def delete_directory_with_contents():
    try:
        shutil.rmtree(data_dir)
        button_create_dataset.configure(state="disabled")
        button_train_classifier.configure(state="disabled")
        button_inference_classifier.configure(state="disabled")
        button_safe_model.configure(state="disabled")
        logging.info(f"Directory '{data_dir}' and its contents deleted.")
    except Exception as e:
        logging.error(f"Error deleting directory '{data_dir}': {e}")


def motor_control_thread(direction_case):
    while not stop_motor.is_set():
        control_motor(direction_case)


def control_conveyor(direction, event_type):
    global motor_thread, stop_motor
    if event_type == 'press':
        stop_motor.clear()
        direction_case = direction == 'forward'
        if motor_thread is None or not motor_thread.is_alive():
            motor_thread = threading.Thread(target=motor_control_thread, args=(direction_case,), daemon=True)
            motor_thread.start()
    elif event_type == 'release':
        stop_motor.set()
        if motor_thread is not None:
            motor_thread.join()
            motor_thread = None


def toggle_auto_mode():
    global is_auto_running, motor_thread, stop_motor
    if not is_auto_running:
        is_auto_running = True
        stop_motor.clear()
        button_conveyor_auto.configure(bg="#FF0000", text="Stop Auto")
        if motor_thread is None or not motor_thread.is_alive():
            motor_thread = threading.Thread(target=motor_control_thread, args=(True,), daemon=True)
            motor_thread.start()
    else:
        is_auto_running = False
        stop_motor.set()
        if motor_thread is not None:
            motor_thread.join()
            motor_thread = None
        button_conveyor_auto.configure(bg="#006400", text="Move Auto")


def on_closing():
    global stop_motor
    stop_motor.set()
    if motor_thread is not None:
        motor_thread.join()
    root.destroy()


# GUI Layout
root.protocol("WM_DELETE_WINDOW", on_closing)

frame = tk.Frame(master=root, bg="#2e3b4e")
frame.pack(pady=20, padx=40, fill="both", expand=True)

title = tk.Label(master=frame, text="Dataset Generator & Conveyor Control", font=("Helvetica", 24, "bold"), bg="#2e3b4e", fg="#ffffff")
title.pack(pady=20, padx=8)

class_entry_label = tk.Label(master=frame, text="Enter class name:", **label_style)
class_entry_label.pack(pady=5, padx=8)

class_entry = tk.Entry(master=frame, font=("Helvetica", 12), width=30)
class_entry.pack(pady=5, padx=8)

add_class_button = tk.Button(master=frame, text="Add Class", command=add_class_entry, **button_style)
add_class_button.pack(pady=10, padx=8)

button_collecting_data = tk.Button(master=frame, text="Collect Data", state="disabled", command=run_collecting_data, **button_style)
button_collecting_data.pack(pady=10, padx=10)

button_create_dataset = tk.Button(master=frame, text="Create Dataset", state="disabled", command=run_create_dataset, **button_style)
button_create_dataset.pack(pady=10, padx=10)

button_train_classifier = tk.Button(master=frame, text="Train Classifier", state="disabled", command=run_train_classifier, **button_style)
button_train_classifier.pack(pady=10, padx=10)

button_inference_classifier = tk.Button(master=frame, text="Inference Classifier", state="disabled", command=run_inference_classifier, **button_style)
button_inference_classifier.pack(pady=10, padx=10)

button_safe_model = tk.Button(master=frame, text="Save Model", state="disabled", command=safe_model_file, **button_style)
button_safe_model.pack(pady=10, padx=10)

button_delete_folder = tk.Button(master=frame, text="Delete folder", command=delete_directory_with_contents, **button_style)
button_delete_folder.pack(pady=10, padx=10)

button_conveyor_forward = tk.Button(master=frame, text="Move Conveyor Forward", **button_style)
button_conveyor_forward.pack(pady=10, padx=10)
button_conveyor_forward.bind('<ButtonPress>', lambda event: control_conveyor('forward', 'press'))
button_conveyor_forward.bind('<ButtonRelease>', lambda event: control_conveyor('forward', 'release'))

button_conveyor_backward = tk.Button(master=frame, text="Move Conveyor Backward", **button_style)
button_conveyor_backward.pack(pady=10, padx=10)
button_conveyor_backward.bind('<ButtonPress>', lambda event: control_conveyor('backward', 'press'))
button_conveyor_backward.bind('<ButtonRelease>', lambda event: control_conveyor('backward', 'release'))

button_conveyor_auto = tk.Button(master=frame, text="Move Auto", bg="#006400", fg="#FFFFFF", font=("Helvetica", 12, "bold"),
                                 command=toggle_auto_mode, relief=tk.RAISED, bd=3, width=20)
button_conveyor_auto.pack(pady=10, padx=10)

root.mainloop()
