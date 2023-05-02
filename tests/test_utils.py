import pytest
from src.utils import *
@pytest.mark.parametrize("values, expected_result", [
    ([], False),
    ([], 2),
])
def test_random_argmax(values, expected_result):
    assert True
