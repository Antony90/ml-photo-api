from typing import Union
from bson import ObjectId
import numpy as np
from pymongo import ASCENDING, HASHED, TEXT, MongoClient, IndexModel, ReturnDocument
from pymongo.client_session import ClientSession
from fastapi import HTTPException
from pymongo.database import Database
from pymongo.errors import OperationFailure
from dotenv import load_dotenv
from os import environ
from .util.models import FaceEncoding, Person
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
        self.db, self.session = self.init_db(local, reset)
        if reset:
            from time import sleep
            print("!! DELETING ALL COLLECTIONS !!")
            self.reset()

    def __del__(self):
        self.session.end_session()
        
    def reset(self):
        'Recreate collections and indexes'
        print("Recreating collections and indexes...", end=" ")
        self.db.drop_collection('users')
        self.db.drop_collection('people')
        self.db.drop_collection('images')
        unique_user_id = IndexModel([("user_id", TEXT)], unique=True)
        self.db.create_collection('users').create_indexes([unique_user_id])
        self.db.create_collection('people')
        unique_image_id = IndexModel([("image_id", TEXT)], unique=True)
        self.db.create_collection('images')
        print("done!")

    def init_db(self, local: bool, reset: bool) -> tuple[Database, ClientSession]:
        # MongoDB Atlas URL to connect pymongo to database
        connection_string = "mongodb://localhost" if local \
            else f"mongodb+srv://{env['MONGODB_USER']}:{env['MONGODB_PASS']}@cluster0.rr5os7q.mongodb.net/{env['DB']}"

        client = MongoClient(connection_string)
        db = client['faces' if local else env['DB']]
        return db, client.start_session()

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
            cnt = self.db.people.update_one({'_id': person_id},
                                      {'$push': {'encodings': {'$each': face_enc_dicts}}}).modified_count

        except OperationFailure as e:
            raise HTTPException(status_code=500, detail=str(e.details))
        else:
            if cnt == 0:
                raise HTTPException(status_code=404, detail=f"Person ID {person_id} not found")

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
            face_encodings = self.db.people.find_one(
                {'_id': person_id})['encodings']
            # Convert face_encoding doc to class instances
            person_doc = self.db.people.find_one({'_id': person_id})
            person = Person(person_id, person_doc['name'])
            people_face_encodings[person] = [
                FaceEncoding.from_dict(face_enc) for face_enc in face_encodings]
        return people_face_encodings

    def set_person_name(self, user_id: str, person_id: str, name: str):
        person_oid = ObjectId(person_id)
        user_doc = self.db.users.find_one({'user_id': user_id}, {'people': 1})
        if user_doc is None:
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")
        if person_oid not in user_doc['people']:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} does not have person with id {person_id}")

        num_updated = self.db.people.update_one(
            {'_id': person_oid}, { '$set': { 'name': name }}).modified_count
        return num_updated == 1

    def delete_user_image(self, user_id: str, image_id: str) -> list[ObjectId]:
        with self.session.start_transaction():
            image_doc = self.db.images.find_one_and_delete(
                {'user_id': user_id, 'image_id': image_id}, {'people': 1})
            if image_doc is None:
                return []
            people: list[ObjectId] = image_doc['people']
            # Remove the image from each person
            for person_id in people:
                remove_img = {'$pull': {'encodings': {'image_id': image_id}}}
                updated_person_doc = self.db.people.find_one_and_update(
                    {'_id': person_id},
                    remove_img,
                    return_document=ReturnDocument.AFTER
                )
                # If the person has no images now, delete their doc from people collection
                # And from the user's people array
                if len(updated_person_doc['encodings']) == 0:
                    self.db.people.delete_one({ '_id': person_id })
                    self.db.users.update_one({'user_id': user_id}, {'$pull': { 'people': person_id }})
                    
            # Returns the person ids of people whose photos were deleted
            # Since a single photo can have multiple faces, multiple people can be updated
        return people 

    def insert_image_person(self, face_encs: list[FaceEncoding], user_id: str, person_id: ObjectId):
        # Store person under each image to make image deletion easier
        for face in face_encs:
            # addToSet ensures no duplicate ids exist in people array
            # Duplicates are expected from 1 photo having two faces matching one person
            # upsert=True will create the image doc if not already existing
            try:
                self.db.images.update_one({'image_id': face.image_id, 'user_id': user_id}, {
                                      '$addToSet': {'people': person_id}}, upsert=True)
            except OperationFailure as err:
                raise HTTPException(status_code=500, details=str(err.details))
            
    def get_user_image_ids(self, user_id) -> list[str]:
        return [ img_doc['image_id'] for img_doc in self.db.images.find({ 'user_id': user_id }) ]
        

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
