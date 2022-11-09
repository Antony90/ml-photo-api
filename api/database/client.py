import numpy as np
from pymongo import ASCENDING, HASHED, MongoClient
from pymongo.database import Database
from dotenv import load_dotenv
from os import environ
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
    def __init__(self, local: bool):
        self.db = self.init_db(local)
        self.encodings = self.db.encodings

    def init_db(self, local: bool) -> Database:
        # MongoDB Atlas URL to connect pymongo to database
        connection_string = "mongodb://localhost" if local else f"mongodb+srv://{env['MONGODB_USER']}:{env['MONGODB_PASS']}@cluster0.rr5os7q.mongodb.net/{env['DB']}"

        client = MongoClient(connection_string)
        db = client['faces' if local else env['DB']]

        if new_database:
            'Create indexes'
        return db

    def insert_encoding(self, user_id: str, person_id: str, encoding: np.ndarray, img_id: str):
        try:
            self.db.encodings.insert_one({
                'user_id': user_id,
                'img_id': img_id,
                'person_id': person_id,
                'encoding': np.array2string(encoding)
            })
        except Exception as e:
            print(e)


if __name__ == '__main__':
    db = FaceDatabase(local=False)
    db.insert_encoding("user_id", "person_id", np.array([1,2,3]), "img_id")
    print(db.encodings.find())
