from collections import namedtuple
import pytest
from raster_compat import compatible_settings

OldSettings = namedtuple('OldSettings', ['width'])

def test_older_rasterizer_accepts_disabled_antialiasing():
    assert compatible_settings(OldSettings,dict(width=640,antialiasing=False)).width==640

def test_unsupported_enabled_feature_is_not_silently_disabled():
    with pytest.raises(TypeError,match='antialiasing'):
        compatible_settings(OldSettings,dict(width=640,antialiasing=True))

def test_other_unknown_options_are_rejected():
    with pytest.raises(TypeError):
        compatible_settings(OldSettings,dict(width=640,unknown=True))
