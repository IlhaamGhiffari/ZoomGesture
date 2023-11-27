import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, ttk
from PIL import Image, ImageTk
from threading import Thread, Lock

class HandTrackingZoomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking and Image Zoom")
        self.lock = Lock()  # To ensure thread-safety
        self.setup_mediapipe()
        self.setup_ui()
        self.initialize_variables()

    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def setup_ui(self):
        self.status_label = ttk.Label(self.root, text="Select an image to start.")
        self.status_label.grid(row=0, columnspan=2, sticky='ew', padx=10, pady=10)

        ttk.Button(self.root, text="Select Image", command=self.select_image).grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        ttk.Button(self.root, text="Exit", command=self.exit_program).grid(row=1, column=1, sticky='ew', padx=10, pady=5)

        self.image_frame = ttk.Frame(self.root)
        self.image_frame.grid(row=2, columnspan=2, sticky='nsew', padx=10, pady=10)

        self.webcam_label = ttk.Label(self.image_frame)
        self.webcam_label.grid(row=0, column=0, sticky='nsew')

        self.zoom_label = ttk.Label(self.image_frame)
        self.zoom_label.grid(row=0, column=1, sticky='nsew')

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)


    def initialize_variables(self):
        self.image = None
        self.zoomed_image = None
        self.initial_distance = None
        self.scale = 1
        self.previous_scale = 1
        self.running = True
        self.cap = None

    def select_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is None:
                self.status_label.config(text="Failed to load image.")
            else:
                self.status_label.config(text="Image loaded successfully.")
                self.initial_distance = None
                self.scale = 1
                self.display_initial_image()
                self.start_webcam()

    def display_initial_image(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.zoom_label.imgtk = imgtk
        self.zoom_label.config(image=imgtk)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Could not open webcam.")
            return
        self.status_label.config(text="Webcam started.")
        Thread(target=self.webcam_feed, daemon=True).start()

    def webcam_feed(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.process_frame(frame)

            self.update_gui(frame)

        self.cap.release()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            if len(results.multi_hand_landmarks) == 2:
                self.update_zoom_scale(results.multi_hand_landmarks)

    def update_zoom_scale(self, hand_landmarks):
        distance = self.calculate_distance(hand_landmarks)
        if self.initial_distance is None:
            self.initial_distance = distance

        new_scale = distance / self.initial_distance
        # Smooth transition for zoom scale
        self.scale += (new_scale - self.scale) * 0.1

        if abs(self.scale - self.previous_scale) > 0.01 and self.image is not None:
            self.update_zoomed_image()
            self.previous_scale = self.scale

    def calculate_distance(self, hand_landmarks, hand_no=0):
        thumb_tip = np.array([hand_landmarks[hand_no].landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                              hand_landmarks[hand_no].landmark[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([hand_landmarks[hand_no].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks[hand_no].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        return np.linalg.norm(index_tip - thumb_tip)

    def update_zoomed_image(self):
        zoomed = cv2.resize(self.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed.shape[:2]
        h, w = self.image.shape[:2]
        top, left = max((zh - h) // 2, 0), max((zw - w) // 2, 0)
        self.zoomed_image = zoomed[top:top+h, left:left+w]

        img = cv2.cvtColor(self.zoomed_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.zoom_label.imgtk = imgtk
        self.zoom_label.config(image=imgtk)

    def exit_program(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def update_gui(self, frame):
        """ Update GUI elements safely from the webcam thread. """
        try:
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_photo = ImageTk.PhotoImage(image=frame_image)
            zoomed_img = Image.fromarray(cv2.cvtColor(self.zoomed_image, cv2.COLOR_BGR2RGB)) if self.zoomed_image is not None else frame_image
            zoomed_imgtk = ImageTk.PhotoImage(image=zoomed_img)

            def update():
                with self.lock:
                    self.webcam_label.imgtk = frame_photo
                    self.webcam_label.config(image=frame_photo)
                    self.zoom_label.imgtk = zoomed_imgtk
                    self.zoom_label.config(image=zoomed_imgtk)

            self.root.after(0, update)
        except Exception as e:
            print("Error updating GUI:", e)

# Create the Tkinter application
root = tk.Tk()
app = HandTrackingZoomApp(root)
root.mainloop()
