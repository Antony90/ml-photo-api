import numpy as np
from face_recognition import face_encodings, compare_faces

from api.util.image import base64_img_to_array

db = None

def get_face_encodings(userID, face_b64_img):
    faces = db.collection('users') \
        .doc(userID) \
        .collection('faces')

    # List of face encodings to compare against
    known_face_encodings = []
    for face in faces:
        encoding = np.array(face['encoding'])
        known_face_encodings.append(encoding)

    img = base64_img_to_array(face_b64_img)

    face_recognition.compare_faces(
      known_face_encodings,
    )
    