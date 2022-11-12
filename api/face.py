from collections import defaultdict
from random import choice

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from face_recognition import (compare_faces,
                              face_encodings, face_landmarks)

from util.image import base64_img_to_array
from util.models import FaceEncoding


def get_face_encodings(images: list[dict[str, str]]) -> list[FaceEncoding]:
    """Converts a base-64 image into np array and
    finds the face encoding output for every face
    in the image.

    Args:
        images (list[dict[str, str]]): List of dictionaries with attributes 
            'image' - base 64 image with faces \n
            'id' - image id

    Returns:
        list[FaceEncoding]: List of face encodings for each face in every image. \n
        All images are assumed to have a face. If not, procedure still exits peacefully.
    """
    face_encodings_list = []
    for img in images:
        img_arr = base64_img_to_array(img['image'])

        # Find face encodings for each face in the image
        encodings = face_encodings(img_arr, model="small")
        assert len(encodings) != 0

        faces = [FaceEncoding(id=img['id'], encoding=enc) for enc in encodings]
        face_encodings_list.extend(faces)

    return face_encodings_list


def match_face_encodings_to_people(
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


def cluster_unmatched_encodings(
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


def has_face(imgs: list[np.ndarray]):
    return [i for i, face_boxes in enumerate([face_landmarks(img) for img in imgs]) if face_boxes is not []]


if __name__ == "__main__":
    enc1 = face_encodings("util/elon-large.jpg")[0]
    enc2 = face_encodings("util/elon-small.jpg")[0]
    faces = [
        FaceEncoding(id='id1', encoding=enc1),
        FaceEncoding(id='id2', encoding=enc2),
    ]
    cluster_unmatched_encodings(faces)
