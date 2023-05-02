import pytest
from src.utils import *
@pytest.mark.parametrize("values, expected_result", [
    ([1, 2, 3, 4], False),
    ([1], 2),
])
def test_random_argmax(values, expected_result):
    result = random_argmax(values)
    assert result <= len(values)
    assert result is not None
