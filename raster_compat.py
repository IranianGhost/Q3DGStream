"""Support rasterizer releases without an optional, disabled antialiasing flag."""
def compatible_settings(settings_type, values):
    values = dict(values)
    if 'antialiasing' not in settings_type._fields and values.get('antialiasing') is False:
        values.pop('antialiasing')
    # The constructor still rejects enabled unsupported features and unknown keys.
    return settings_type(**values)
