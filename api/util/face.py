from random import choice

import numpy as np
from face_recognition import (batch_face_locations, compare_faces,
                              face_encodings)

from api.util.image import base64_img_to_array

db = None


def get_face_encodings(b64_imgs: list[str]) -> list[list[np.ndarray]]:
    """Converts a base-64 image into np array and
    finds the face encoding output for every face
    in the image.

    Args:
        b64_imgs (list[str]): List of base 64 images

    Returns:
        list[list[np.ndarray]]: List of face encodings for each image. \n
        If an image has no face, its entry in the outer list is an empty list.
    """
    img_encodings = []
    for b64_img in b64_imgs:
        img = base64_img_to_array(b64_img)

        # One encoding for each face in the image
        encodings = face_encodings(img)
        img_encodings.append(encodings)

    return img_encodings


def match_img_encodings_to_people(
    encodings: list[np.ndarray],
    people: dict[int, list[np.ndarray]]
) -> tuple[list[np.ndarray], dict[int, list[np.ndarray]]]:
    """
    Matches a list of encodings of faces from an image 
    to the people they relate to. This involves calculating
    the distance between each encoding of a person and each 
    encoding in `encodings`. If below a threshold,
    the encoding is added to the person's encoding list.

    Some encodings may be unmatched, and are collected as the
    first element in the returned tuple.

    Args:
        encodings (list[np.ndarray]): 
            List of face encodings for each face in an image.
        people: (dict[int, list[np.ndarray]])): 
            Mapping from person id to list of face encodings which are similar to each other.

    Returns:
        `tuple[list[np.ndarray], dict[int, list[np.ndarray]]]`: Tuple of unmatched encodings and updated mapping from person id to similar encodings.
    """
    # Initially all encodings are unmatched
    matched_encs = [False for _ in range(len(encodings))]
    for person_id, person_encs in people.items():
        # Since `encodings` comes from one image (with at least one detected face),
        # if any encoding from `encodings` matches any encoding
        # person_encoding in `person_encs`, add the `encodings` to the
        # list under `person_id` dict key
        for i, enc in enumerate(encodings):
            if any(compare_faces(person_encs, enc)):
                person_encs.append(enc)
                # Mark encoding as being matched to a person
                matched_encs[i] = True
        people[person_id] = person_encs

    unmatched_encodings = [encodings[i] for (
        i, is_matched) in enumerate(matched_encs) if not is_matched]

    return unmatched_encodings, people


def handle_unmatched_encodings(
    encodings: list[np.ndarray], 
    people: dict[int, list[np.ndarray]]
) -> dict[int, list[np.ndarray]]:
    """These encodings do not match any person's face.
    They must be the encodings of `len(encodings)` unique faces
    since similar encodings in the same image are already removed.
    
    Args:
        encodings: Face encodings which do not match any person's face
        people: dict mapping person to a list of similar face encodings
            
    Returns:
        Updated person-encoding list mappings where each unmatched encoding is assigned to a new person"""
    next_person_id = max(people.keys()) + 1
    for i in range(len(encodings)):
        # Create a mapping from a new person to a singleton encoding
        people[next_person_id + i] = [encodings[i]]
        
    return people


"""
User {
    id: string
}

`encodings` collection has documents:
Encoding {
    user_id: string,
    id: string (photo id)
    person_id: references Person(id),
    encoding: binary
}

`people` collection has documents:
Person {
    id: string,
    user_id: string references User(id)
}
"""
