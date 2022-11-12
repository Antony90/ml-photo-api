from typing import Union
from bson import ObjectId
import numpy as np
from pymongo import ASCENDING, HASHED, TEXT, MongoClient, IndexModel
from werkzeug.exceptions import InternalServerError
from pymongo.database import Database
from dotenv import load_dotenv
from os import environ
from util.models import FaceEncoding, Person
env = environ
load_dotenv()

# Flag for when database is new
# Will create index for unique article links
new_database = False
local = True

"""
Face encoding data is stored in a local NoSQL MongoDB database
`users` collection has documents:
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


class FaceDatabase:
    def __init__(self, local: bool, reset: bool = False):
        if reset:
            from time import sleep
            print("!! DELETING ALL COLLECTIONS 5 SECONDS TO CANCEL !!")
            sleep(5)
        self.db = self.init_db(local, reset)

    def init_db(self, local: bool, reset: bool) -> Database:
        # MongoDB Atlas URL to connect pymongo to database
        connection_string = "mongodb://localhost" if local \
            else f"mongodb+srv://{env['MONGODB_USER']}:{env['MONGODB_PASS']}@cluster0.rr5os7q.mongodb.net/{env['DB']}"

        client = MongoClient(connection_string)
        db = client['faces' if local else env['DB']]

        if reset:
            'Recreate collections and indexes'
            print("Recreating collections and indexes...", end=" ")
            db.drop_collection('users')
            db.drop_collection('people')
            unique_user_id = IndexModel([("user_id", TEXT)], unique=True)
            db.create_collection('users').create_indexes([unique_user_id])
            db.create_collection('people')
            print("done!")
        return db

    def create_user(self, user_id: str):
        return self.db.users.insert_one({'user_id': user_id, 'people': []}).inserted_id

    def add_person_to_user(self, user_id: str, person_id: ObjectId) -> bool:
        '''Inserts a person under the specified user. 
        Each user has an array of `person_id`s where each
        person has a list of face encodings.

        Returns true if user's people array was updated.
        '''
        return self.db.users.update_one(
            {'user_id': user_id},
            {'$push': {'people': person_id}}
        ).modified_count == 1

    def create_person(self, name: str) -> ObjectId:
        '''
        Create a new person and return its unique id.
        '''
        return self.db.people.insert_one({'name': name, 'encodings': []}).inserted_id

    def insert_encodings(self, person_id: ObjectId, face_encs: list[FaceEncoding]):
        '''
        Updates a person's face encodings by appending a list of new encodings.
        Each dictionary in `images` contains attributes `encoding` and `id`.
        '''
        # Convert np encoding to string before inserting
        face_enc_dicts = [face_enc.to_dict() for face_enc in face_encs]
        try:
            self.db.people.update_one({'_id': person_id},
                                      {'$push': {'encodings': {'$each': face_enc_dicts}}})
        except Exception as e:
            print(e)

    def get_user_face_encodings(self, user_id: str) -> dict[Person, list[FaceEncoding]]:
        '''
        Given a user id, return a mapping of `person_id` to their face `img_encodings`.
        Each `img_encoding` is an object with `img` and `encoding` attributes/keys.
        '''
        user_doc = self.db.users.find_one({'user_id': user_id})
        if user_doc is None:
            return None
        people_face_encodings = dict()

        for person_id in user_doc['people']:
            try:
                face_encodings = self.db.people.find_one(
                    {'_id': person_id})['encodings']
            except Exception as e:
                return print(e)
            else:
                # Convert face_encoding doc to class instances
                person_doc = self.db.people.find_one({ '_id': person_id })
                person = Person(person_id, person_doc['name'])
                people_face_encodings[person] = [
                    FaceEncoding.from_dict(face_enc) for face_enc in face_encodings]
        return people_face_encodings

    def set_person_name(self, user_id: str, person_id: str, name: str):
        user_doc = self.db.users.find_one({'user_id': user_id}, {'people': 1})
        people = user_doc['people']
        print(people)
        if person_id not in people:
            raise ValueError(
                f"User {user_id} does not have person with id {person_id}")
            
        num_updated = self.db.people.update_one({'_id': person_id}, {'name': name}).modified_count
        return num_updated == 1

    def delete_user_image(self, user_id: str, person_id: str, image_id: str):
        person_oid = ObjectId(person_id)
        user_doc = self.db.users.find_one({'user_id': user_id}, {'people': 1})
        people = user_doc['people']
        if person_oid not in people:
            raise ValueError(
                f"User {user_id} does not have person with id {person_id}")

        remove_img = { '$pull':
            { 'encodings': { 'image_id': image_id } }
        }
        num_updated = self.db.people.update_one({'_id': person_oid}, remove_img).modified_count
        return num_updated == 1


if __name__ == '__main__':
    from face_recognition import face_encodings, load_image_file

    db = FaceDatabase(local=False, reset=True)
    test_user_id = "firebase_user_id2"
    img = load_image_file('./api/test.jpg')
    encoding = face_encodings(img)[0]

    db.create_user(test_user_id)
    print("Created user")
    person_id = db.create_person('Elon Musk')
    print(f"Created {person_id=}")
    db.add_person_to_user(test_user_id, person_id)
    print("Added person to user")

    db.insert_encodings(person_id, [FaceEncoding.from_dict({
        'encoding': str(list(encoding)),
        'id': 'firebase_img_id1'
    }), FaceEncoding.from_dict({
        'encoding': str(list(encoding)),
        'id': 'firebase_img_id2'
    })
    ])
    print("Pushed new image encodings")

    people_face_encodings = db.get_user_face_encodings(test_user_id)
    print(people_face_encodings)
