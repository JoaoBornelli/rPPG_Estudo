"""
Captura de webcam + MediaPipe Face Landmarker.

Encapsula:
- VideoCapture loop (next_frame → (rgb, landmarks, timestamp_ms))
- desenho dos landmarks (mesh, iris)
- helpers de landmark→pixel
"""
import cv2 as cv
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles


def get_landmark_px(face_landmarks, idx, w, h):
    lm = face_landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def landmark_points(face_landmarks, idx_list, w, h):
    pts = []
    for idx in idx_list:
        lm = face_landmarks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if not detection_result.face_landmarks:
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return annotated_image


FRAME_DURATION_MS = 33  # 30 fps


class Capture:
    def __init__(self, model_path="face_landmarker.task", source=0):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.cap = cv.VideoCapture(source)
        self.timestamp_ms = 0

    def next_frame(self):
        """
        Retorna (rgb_frame, detection_result, timestamp_ms) ou None se câmera falhar.
        Avança timestamp_ms em FRAME_DURATION_MS.
        """
        ret, frame_bgr = self.cap.read()
        if not ret:
            return None
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        ts = self.timestamp_ms
        self.timestamp_ms += FRAME_DURATION_MS
        return rgb, detection, ts

    def release(self):
        self.cap.release()
