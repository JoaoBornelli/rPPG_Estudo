import cv2 as cv
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# =====================================================
# Função para desenhar os landmarks
# =====================================================
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if not detection_result.face_landmarks:
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:

        # Malha facial (triângulos)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Contornos do rosto
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )

        # Íris esquerda
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

        # Íris direita
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image


# =====================================================
# Inicialização do FaceLandmarker (MODO VÍDEO)
# =====================================================
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # ⚠️ ESSENCIAL
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

detector = vision.FaceLandmarker.create_from_options(options)


# =====================================================
# Captura de vídeo (arquivo ou webcam)
# =====================================================
# Para vídeo:
cap = cv.VideoCapture("media/video.mp4")

# Para webcam, use:
# cap = cv.VideoCapture(0)

timestamp_ms = 0
frame_duration_ms = 33  # ~30 FPS


# =====================================================
# Loop principal
# =====================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV trabalha em BGR | MediaPipe em RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    annotated_frame = draw_landmarks_on_image(
        rgb_frame,
        detection_result
    )

    cv.imshow(
        "Face Mesh Video",
        cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)
    )

    timestamp_ms += frame_duration_ms

    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break


# =====================================================
# Finalização
# =====================================================
cap.release()
cv.destroyAllWindows()
