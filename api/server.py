from flask import Flask, Response, request
from flask_cors import CORS
from werkzeug.exceptions import InternalServerError
from pymongo.errors import BulkWriteError

from classify import ImageSceneClassifier
from util.image import images_to_arrays
from face import cluster_unmatched_encodings, get_face_encodings, has_face, match_face_encodings_to_people, to_person_img_ids
from face_db import FaceDatabase

app = Flask(__name__)
CORS(app)

# load model and category data
classifier = ImageSceneClassifier()
face_db = FaceDatabase(local=False, reset=False)


@app.route('/faces/<user_id>/process', methods=['POST'])
def process_faces(user_id):
    """
    This route is provided an array of base64 images each containing at least one
    face. Each face, if recognized: is assigned to one of the user's existing people.
    Otherwise, it is assigned to a new person based on similarity to other
    unrecognized faces using Hierarchical Clustering to group such face enocdings
    together.
    """
    data = request.get_json()

    try:
        images = data.get("images")
    except KeyError:
        return "Missing parameter 'images'", 400

    people_face_encodings = face_db.get_user_face_encodings(user_id)
    print("Fetched stored faces")
    if people_face_encodings is None:
        try:
            face_db.create_user(user_id)
        except Exception as exception:
            return str(exception), 500
        else:
            print(f"Created user {user_id}")
            people_face_encodings = dict()

    num_people = len(people_face_encodings.keys())

    faces = get_face_encodings(images)
    print("Got face encodings")
    unmatched_faces, updated_people_faces = match_face_encodings_to_people(
        faces, people_face_encodings)
    print("Matched face encodings")
    new_people_faces = cluster_unmatched_encodings(unmatched_faces, num_people)
    print("Clustered")

    # For each existing person, append their newly matched face encodings in the database
    for person, face_encodings in updated_people_faces.items():
        print(person)
        try:
            face_db.insert_encodings(person.id, face_encodings)
        except BulkWriteError as exc:
            return exc.details['writeErrors'][0]['errmsg'], 400


    # For each new person, create a Person document, add it to the user's people array
    # and insert the person's encodings.
    for person_name, face_encodings in new_people_faces.items():
        print(person_name)
        person_id = face_db.create_person(person_name)
        face_db.add_person_to_user(user_id, person_id)
        try:
            face_db.insert_encodings(person_id, face_encodings)
        except BulkWriteError as exc:
            return exc.details['writeErrors'][0]['errmsg'], 400
        
    return Response(status=200)


@app.route('/faces/<user_id>', methods=['GET'])
def get_faces(user_id):
    people_faces = face_db.get_user_face_encodings(user_id)
    return { # Remove encoding data
        'people': to_person_img_ids(people_faces)
    }, 200


@app.route('/faces/<user_id>/<person_id>/rename', methods=['PATCH'])
def rename_person(user_id, person_id):
    try:
        name = request.get_json().get("name")
    except KeyError:
        return "Missing parameter 'name'", 400
    try:
        face_db.set_person_name(user_id, person_id, name)
    except Exception as excpt:
        return excpt.args[0], 500
    else:
        return Response(status=200)


@app.route('/faces/<user_id>/<person_id>/<image_id>', methods=['DELETE'])
def delete_person_img(user_id, person_id, image_id):
    try:
        deleted = face_db.delete_user_image(user_id, person_id, image_id)
    except ValueError as err:
        return err.args[0], 404
    else:
        if not deleted:
            return f"Image not found under person {person_id}" , 404
        else:
            return Response(status=200)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()

    try:
        # List of images encoded in base64
        base64_imgs = data.get("images")
    except KeyError:
        return "Missing parameter 'images'", 400

    if not type(base64_imgs) == list:
        return "Bad format: 'images' parameter must be a list", 400

    if len(base64_imgs) == 0:
        return "Empty image array", 400

    # Convert images from encoded base64 to np array format with shape (160, 160)
    try:
        img_batch = images_to_arrays(base64_imgs)
    except InternalServerError as e:
        return e.description, 400

    # Get indexes of images with faces
    face_idxs = has_face(img_batch)

    predictions = classifier.predict(img_batch)
    image_tags = classifier.tags_from_predictions(predictions)
    results = [
        {'tags': image_tags[i],
         'face': True if i in face_idxs else False}
        for i in range(len(img_batch))]
    print(f'{image_tags=}')
    return {"results": results}, 200


if __name__ == '__main__':
    app.run(debug=True)
