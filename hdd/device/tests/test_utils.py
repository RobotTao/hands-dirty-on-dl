import pytest

from hdd.device.utils import get_device


def test_get_device():
    with pytest.raises(Exception):
        get_device("does-not-exit-device")
    assert get_device("cpu") is not None
