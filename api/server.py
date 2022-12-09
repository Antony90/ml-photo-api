from fastapi import Body, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware

from .classify import ImageSceneClassifier
from .face import (cluster_unmatched_encodings, get_face_encodings, has_face,
                   match_face_encodings_to_people, to_person_img_ids)
from .face_db import FaceDatabase
from .util.image import images_to_arrays
from .util.models import ClassifyResult, Image, PersonFaces
from .docs import options


app = FastAPI(**options)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

success = lambda: {'msg': 'Success'}


# load model and category data
classifier = ImageSceneClassifier()
face_db = FaceDatabase(local=False, reset=True)

@app.post('/reset')
def delete():
    face_db.reset()

@app.post('/faces/{user_id}/process', 
          status_code=status.HTTP_201_CREATED, tags=['Face'], 
          response_description="Number of faces processed")
def process_faces(images: list[Image] = Body(description="List of images in base 64 format and their ID"),
                  user_id: str = Path(title="User ID of user to match group faces for")):
    """
    Groups an array of images with faces by feature similarity. Similar faces are grouped under an abstract Person.
    Each face, if recognized: is assigned to one of the user's existing people.
    Otherwise, it is assigned to a new person based on similarity to other
    unrecognized faces using Hierarchical Clustering to group such face enocdings
    together.
    """
    image_ids = [image.id for image in images]
    stored_ids = face_db.get_user_image_ids(user_id)
    existing_ids = set(image_ids).intersection(set(stored_ids))
    if len(existing_ids) != 0:
        raise HTTPException(detail=f"Images with IDs {existing_ids} already exist in database",
                            status_code=400)

    people_face_encodings = face_db.get_user_face_encodings(user_id)
    if people_face_encodings is None:
        try:
            face_db.create_user(user_id)
        except Exception as exception:
            return str(exception), 500
        else:
            people_face_encodings = dict()

    num_people = len(people_face_encodings.keys())

    faces = get_face_encodings(images)
    unmatched_faces, updated_people_faces = match_face_encodings_to_people(
        faces, people_face_encodings)
    if len(unmatched_faces) == 1:
        # If only one face is unmatched, assign it to a new person
        new_people_faces = { f'Person {num_people + 1}': faces } 
    else:
        # Otherwise use clustering to group faces by similarity
        new_people_faces = cluster_unmatched_encodings(unmatched_faces, num_people)

    # For each existing person, append their newly matched face encodings in the database
    for person, face_encodings in updated_people_faces.items():
        face_db.insert_encodings(person.id, face_encodings)
        # Update link from image to person
        face_db.insert_image_person(face_encodings, user_id, person.id)

    # For each new person, create a Person document, add it to the user's people array
    # and insert the person's encodings.
    for person_name, face_encodings in new_people_faces.items():
        person_id = face_db.create_person(person_name)
        face_db.add_person_to_user(user_id, person_id)
        face_db.insert_encodings(person_id, face_encodings)
        # Update link from image to person
        face_db.insert_image_person(face_encodings, user_id, person_id)

    return len(faces) # number of faces detected

@app.get('/faces/{user_id}', response_model=list[PersonFaces], tags=['Face'])
def get_faces(user_id: str = Path(title="User ID to find person-faces mappings for")):
    people_faces = face_db.get_user_face_encodings(user_id)
    if people_faces is None: 
        return []
    # Remove encoding data
    return to_person_img_ids(people_faces)


@app.patch('/faces/{user_id}/{person_id}/rename', tags=['Face'])
def rename_person(name: str,
                  user_id: str = Path(
                      title="User ID of user who with Person ID as one of their known people"),
                  person_id: str = Path(title="ID of person to rename")):
    face_db.set_person_name(user_id, person_id, name)
    return success()


@app.delete('/faces/{user_id}/{image_id}', response_model=list[str], tags=['Face'])
def delete_person_img(user_id: str = Path(title="ID of user who with Person ID as one of their known people"),
                      image_id: str = Path(title="ID of image to delete")):
    """For each person whose face is in the image, delete their reference to the image.
    Returns a list of affected people IDs."""
    affected_people = face_db.delete_user_image(user_id, image_id)
    if affected_people == []:
        raise HTTPException(
            status_code=404, detail=f"User or Image ID not found (0 deletions)")
    # Convert ObjectId to string
    return [str(id) for id in affected_people]


@app.post('/classify', tags=['Scene Classification'], 
          response_model=list[ClassifyResult], 
          response_description="Array of tags and whether an image has a face for each input image in order")
def classify(images: list[str] = Body(title="List of base 64 encoded images to classify tags for")):
    num_imgs = len(images)
    if num_imgs == 0:
        return HTTPException(detail="Empty image array", status_code=400)

    # Convert images from encoded base64 to np array format with shape (160, 160)
    img_batch = images_to_arrays(images)

    # Get indexes of images with faces
    face_idxs = has_face(img_batch)

    predictions = classifier.predict(img_batch)
    image_tags = classifier.tags_from_predictions(predictions)
    print(f'{image_tags=}')

    return [ClassifyResult(
        tags=image_tags[i],
        has_face=True if i in face_idxs else False
    ) for i in range(num_imgs)]
