import cv2
import time
from pathlib import Path
import json
import logging

def setup_logger():
    """Initialisiert den Logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def initialize_camera(cam_id):
    """Initialisiert die Kamera mit der angegebenen ID."""
    logging.info(f"Attempting to initialize camera with ID {cam_id}...")
    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}!")
    logging.info(f"Camera {cam_id} initialized successfully.")
    return cam

def create_data_directory(data_dir):
    """Erstellt das Verzeichnis für die Bilddaten."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Data directory created at: {data_dir}")
    return data_dir

def write_info_file(data_dir, cam_id, class_names):
    """Schreibt die Informationen zu den Klassen und der Kamera in eine JSON-Datei."""
    labels = {i: class_name for i, class_name in enumerate(class_names)}
    info = {'labels': labels, 'task': 'object_detection', 'cam_id': cam_id}
    info_path = data_dir / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f)
    logging.info(f"Info file written to: {info_path}")
    return labels

def calculate_initial_grey_value(cam, box):
    """Berechnet den durchschnittlichen Grauwert der ersten 5 Frames in der ROI."""
    grey_values = []
    for _ in range(5):
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Failed to grab initial frames.")
        roi = frame[box[1]:box[3], box[0]:box[2]]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grey_values.append(gray_roi.mean())
    average_grey = sum(grey_values) / len(grey_values)
    logging.info(f"Initial average grey value: {average_grey}")
    return average_grey

def capture_images(cam, labels, data_dir, grey_interval):
    """Erfasst Bilder basierend auf der Grauwertänderung in der ROI."""
    cv2.namedWindow("Image Collector")
    ready_for_new_detection = True
    last_detection_time = 0

    # Define ROI (Region of Interest)
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to grab frame.")
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    box_size = 50
    box = (center_x - box_size, center_y - box_size, center_x + box_size, center_y + box_size)

    # Berechne den anfänglichen Grauwert
    initial_grey_value = calculate_initial_grey_value(cam, box)
    lower_threshold = initial_grey_value - grey_interval
    upper_threshold = initial_grey_value + grey_interval

    for class_id, class_name in labels.items():
        class_dir = data_dir / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Collecting data for class '{class_name}'...")

        img_counter = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                logging.error("Failed to grab frame.")
                break

            roi = frame[box[1]:box[3], box[0]:box[2]]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray_roi.mean()

            if (mean_brightness < lower_threshold or mean_brightness > upper_threshold) and ready_for_new_detection:
                img_name = f'{img_counter:03d}.jpg'
                img_path = class_dir / img_name
                cv2.imwrite(str(img_path), frame)
                logging.info(f"Image saved: {img_path}")
                img_counter += 1
                ready_for_new_detection = False
                last_detection_time = time.time()

            if not ready_for_new_detection and time.time() - last_detection_time > 0.5:
                ready_for_new_detection = True

            # Anzeige der ROI und Informationen
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {class_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)
            cv2.putText(frame, f'Grey: {mean_brightness:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Image Collector", frame)

            # ESC-Taste zum Wechseln der Klasse
            if cv2.waitKey(1) % 256 == 27:
                logging.info(f"Switching to next class '{class_name}'.")
                break

    cv2.destroyAllWindows()
    logging.info("Image collection completed.")

def main(cam_id, class_names, data_dir, grey_interval=0.1*255):
    """Hauptfunktion für die Bilderfassung."""
    setup_logger()
    data_dir = create_data_directory(data_dir)
    labels = write_info_file(data_dir, cam_id, class_names)

    cam = None
    try:
        cam = initialize_camera(cam_id)
        capture_images(cam, labels, data_dir, grey_interval)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if cam is not None and cam.isOpened():
            cam.release()
            logging.info("Camera released.")
        cv2.destroyAllWindows()
        logging.info("All OpenCV windows closed.")
