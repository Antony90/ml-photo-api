import json
from dataclasses import dataclass
from typing import Union

import numpy as np
from bson import ObjectId
from pydantic import BaseModel, Field


# Used excludively as a schema for api responses
class PersonFaces(BaseModel):
    name: str = Field(description="Name of person, defaults to 'Person X'")
    id: str = Field(description="ID of the person")
    image_ids: list[str] = Field(description="IDs of images which match this person")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "657025ad29e5e30a76e85a3f",
                "name": "Person 1",
                "image_ids": [
                    "5kx3a233lv",
                    "04gja5lcp9"
                ]
            }
        }
        
class Image(BaseModel):
    data: str = Field(description="Image encoded in base64 format, containing at least one face")
    id: str = Field(description="(external) ID of the image")

class ClassifyResult(BaseModel):
    tags: list[str] = Field(description="Tags assigned to the corresponding input image")
    has_face: bool = Field(description="Whether the corresponding input image has a face")
    class Config:
        schema_extra = {
            "example": {
                "tags": ['Family', 'Landmark', 'Building'],
                "has_face": True
            }
        }

@dataclass
class FaceEncoding:
    image_id: str
    encoding: np.ndarray

    def to_dict(self) -> dict[str, Union[str, np.ndarray]]:
        return {'image_id': self.image_id, 'encoding': str(list(self.encoding))}

    @staticmethod
    def from_dict(d: dict[str, str]):
        assert set(d.keys()).intersection(
            {'image_id', 'encoding'}) == {'image_id', 'encoding'}
        d['encoding'] = np.array(json.loads(d['encoding']))
        return FaceEncoding(**d)


@dataclass
class Person:
    id: ObjectId
    name: str

    def __hash__(self):
        return hash(id)


if __name__ == '__main__':
    FaceEncoding.from_dict(
        {'image_id': 'abc', 'encoding': '[1 2]', 'unexpected_kwarg': 'arg'})
