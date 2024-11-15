import pytest
from terrier.evaluate import build_map



def test_empty_string():
    # Test case: empty string
    assert build_map("") == {}

def test_single_key_value_pair():
    # Test case: single key-value pair
    assert build_map("key1=value1") == {"key1": "value1"}

def test_multiple_key_value_pairs():
    # Test case: multiple key-value pairs
    assert build_map("key1=value1,key2=value2,key3=value3") == {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }

def test_key_without_value():
    # Test case: key without a value
    with pytest.raises(IndexError):
        build_map("key1=value1,key2")


def test_whitespace_handling():
    # Test case: whitespace handling
    assert build_map(" key1 = value1 , key2 = value2 ") == {
        " key1 ": " value1 ",
        " key2 ": " value2 ",
    }
