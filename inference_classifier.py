import pickle
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
from neuralnet_features import get_feature_vec


def start_live_view(cam_id, model, labels, get_feature_vec):
    """
    Startet die Live-Ansicht der Kamera mit OpenCV.
    Zeigt das vollständige Kamerabild an.
    """
    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}!")

    # Kameraauflösung ermitteln
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    # Background-Feature erfassen
    ret, background_frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture background image!")
    background_img = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
    background_feature = get_feature_vec(background_img)
    print("Background feature captured.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame.")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feature_vec = get_feature_vec(img)

        predicted_label = 'Background'
        if feature_vec is not None:
            similarity = cosine_similarity([feature_vec], [background_feature])[0][0]
            if similarity > 0.95:
                predicted_label = 'Background'
            else:
                prediction = model.predict(feature_vec[:, np.newaxis].T)
                predicted_label = labels.get(prediction[0], 'Unknown')

        # Text auf das Bild schreiben
        cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Bild anzeigen - skaliert auf vollständige Größe
        cv2.imshow('Live View', frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC-Taste
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()


def start_live_view_tkinter(cam_id, model, labels, get_feature_vec, canvas, root):
    """
    Startet die Live-Ansicht in einem tkinter-Canvas.
    Zeigt das vollständige Kamerabild an.
    """
    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}!")

    # Kameraauflösung ermitteln
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    canvas.config(width=width, height=height)

    ret, background_frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture background image!")
    background_img = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
    background_feature = get_feature_vec(background_img)
    print("Background feature captured.")

    def update_frame():
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame.")
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feature_vec = get_feature_vec(img)

        predicted_label = 'Background'
        if feature_vec is not None:
            similarity = cosine_similarity([feature_vec], [background_feature])[0][0]
            if similarity > 0.95:
                predicted_label = 'Background'
            else:
                prediction = model.predict(feature_vec[:, np.newaxis].T)
                predicted_label = labels.get(prediction[0], 'Unknown')

        # Text auf das Bild schreiben
        cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # OpenCV-Bild in tkinter-kompatibles Format skalieren
        img = Image.fromarray(frame)


        img = img.resize((width, height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)

        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
        root.after(10, update_frame)

    update_frame()


def inference(data_dir, canvas=None, root=None):
    """
    Lädt das Modell und startet die Live-Ansicht entweder mit OpenCV oder tkinter.
    """
    try:
        model_dict = pickle.load(open(data_dir / 'model.pickle', 'rb'))
        model = model_dict['model']

        with open(data_dir / 'info.json', 'r') as f:
            info = json.load(f)
            labels_dict = info['labels']
            cam_id = info['cam_id']

        if canvas and root:
            start_live_view_tkinter(cam_id, model, labels_dict, get_feature_vec, canvas, root)
        else:
            start_live_view(cam_id, model, labels_dict, get_feature_vec)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == '__main__':
    from tkinter import Tk, Canvas
    data_dir = Path('./data')

    choice = input("Choose interface (1: OpenCV, 2: tkinter): ")
    if choice == "2":
        root = Tk()
        root.title("Live Inference")
        canvas = Canvas(root)
        canvas.pack()

        inference(data_dir, canvas, root)
        root.mainloop()
    else:
        inference(data_dir)
