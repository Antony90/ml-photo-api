from flask import Flask, request
from flask_cors import CORS
from werkzeug.exceptions import InternalServerError

from classify import ImageSceneClassifier
from util.image import images_to_arrays
from face import cluster_unmatched_encodings, get_face_encodings, has_face, match_face_encodings_to_people
from face_db import FaceDatabase

app = Flask(__name__)
CORS(app)

# load model and category data
classifier = ImageSceneClassifier()
face_db = FaceDatabase(local=False, reset=False)


@app.route('/faces/update', methods=['POST'])
def faces():
    data = request.get_json()

    try:
        user_id = data.get("user_id")
        images = data.get("images")
    except KeyError:
        return "Missing parameter 'user_id'", 400

    people_face_encodings = face_db.get_user_face_encodings(user_id)
    print("Fetched stored faces")
    if people_face_encodings is None:
        try:
            face_db.create_user(user_id)
        except Exception as e:
            return str(e), 500
        else:
            print(f"Created user {user_id}")
            people_face_encodings = dict()

    print(f"{people_face_encodings=}")
    num_people = len(people_face_encodings.keys())

    faces = get_face_encodings(images)
    print("Got face encodings")
    unmatched_faces, updated_people_faces = match_face_encodings_to_people(
        faces, people_face_encodings)
    print("Matched face encodings")
    new_people_faces = cluster_unmatched_encodings(unmatched_faces, num_people)
    print("Clustered")
    updated_people_faces.update(new_people_faces)
    
    return {
        'people': [
            {
                'name': person,
                'ids': [enc.id for enc in encodings]
            }
        for person, encodings in updated_people_faces.items()]
    }, 200


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
