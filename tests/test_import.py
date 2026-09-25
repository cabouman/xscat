import xscat


def test_version():
    assert isinstance(xscat.__version__, str)
    assert xscat.__version__
