from dataclasses import dataclass
import json
from typing import Union
import numpy as np


@dataclass
class FaceEncoding:
    id: str
    encoding: np.ndarray

    def to_dict(self) -> dict[str, Union[str, np.ndarray]]:
        return {'id': self.id, 'encoding': str(list(self.encoding))}

    @staticmethod
    def from_dict(d: dict[str, str]):
        assert set(d.keys()).intersection({'id', 'encoding'}) == {'id', 'encoding'}
        d['encoding'] = np.array(json.loads(d['encoding']))
        return FaceEncoding(**d)


if __name__ == '__main__':
    FaceEncoding.from_dict(
        {'id': 'abc', 'encoding': '[1 2]', 'unexpected_kwarg': 'arg'})
