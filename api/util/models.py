import json
from dataclasses import dataclass
from typing import Union
from bson import ObjectId

import numpy as np


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
