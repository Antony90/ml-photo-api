from collections import defaultdict
from random import choice

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from face_recognition import (compare_faces,
                              face_encodings)

from .image import base64_img_to_array
from .models import FaceEncoding

db = None


def get_face_encodings(b64_imgs: list[str]) -> list[list[np.ndarray]]:
    """Converts a base-64 image into np array and
    finds the face encoding output for every face
    in the image.

    Args:
        b64_imgs (list[str]): List of base 64 images with faces

    Returns:
        list[list[np.ndarray]]: List of face encodings for each image. \n
        If an image has no face, its entry in the outer list is an empty list.
    """
    img_encodings = []
    for b64_img in b64_imgs:
        img = base64_img_to_array(b64_img)

        # One encoding for each face in the image
        encodings = face_encodings(img, model="small")
        assert len(encodings) != 0
        img_encodings.append(encodings)

    return img_encodings


def match_img_encodings_to_people(
    face_encodings: list[FaceEncoding],
    people: dict[str, list[FaceEncoding]]
) -> tuple[list[FaceEncoding], dict[str, list[FaceEncoding]]]:
    """
    Matches a list of encodings of faces from an image 
    to the people they relate to. This involves calculating
    the distance between each encoding of a person and each 
    encoding in `encodings`. If below a threshold,
    the encoding is added to the person's face list.

    Some encodings may be unmatched, and are collected as the
    first element in the returned tuple. These are later used
    to create a new person/several new people.

    Args:
        face_encodings (list[FaceEncoding]): 
            List of face encodings to be matched.
        people: (dict[str, list[FaceEncoding]]): 
            Mapping from person id to list of face encodings which are similar to each other.

    Returns:
        `tuple[list[FaceEncoding], dict[str, list[FaceEncoding]]]`: 
            Tuple of unmatched encodings and updated mapping from person id to their face encodings.
    """
    # Initially all encodings are unmatched
    matched_faces = [False for _ in range(len(face_encodings))]
    new_people_faces = defaultdict(list)
    for person_id, person_faces in people.items():
        # Since `encodings` comes from one image (with at least one detected face),
        # if any encoding from `encodings` matches any encoding
        # person_encoding in `person_encs`, add the `encodings` to the
        # list under `person_id` dict key
        for i, face in enumerate(face_encodings):
            person_encodings = [face.encoding for face in person_faces]
            # If enc matches any face encodings for the person, add to new_people_faces dict
            if any(compare_faces(person_encodings, face.encoding)):
                new_people_faces[person_id].append(face)
            else:
                # Mark encoding as being matched to a person
                matched_faces[i] = True

    # These encodings will create a new person/several new people
    unmatched_faces = [face_encodings[i] for (
        i, is_matched) in enumerate(matched_faces) if not is_matched]

    return unmatched_faces, new_people_faces


def handle_unmatched_encodings(
    faces: list[FaceEncoding],
    num_existing_people: int
) -> dict[str, list[FaceEncoding]]:
    """These encodings do not match any existing person's face.
    Group them using hierarchical clustering with Euclidean distance
    as a metric and a distance threshold of 0.6.

    Args:
        num_existing_people: 
            int number of people already in database, used to assign numbers to new people
    """
    samples = np.array([face.encoding for face in faces])
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="euclidean",
        distance_threshold=0.6
    ).fit(samples)

    # Grouped images represent a single person
    people = clustering.labels_
    people_faces = defaultdict(list)
    for i, person_num in enumerate(people):
        name = f'Person {person_num + num_existing_people + 1}'
        people_faces[name].append(faces[i])
    return people_faces


if __name__ == "__main__":
    enc1 = face_encodings("./api/util/elon-large.jpg")[0]
    enc2 = face_encodings("./api/util/elon-small.jpg")[0]
    faces = [
        FaceEncoding(id='id1', encoding=enc1),
        FaceEncoding(id='id2', encoding=enc2),
    ]
    handle_unmatched_encodings(faces)
