import ephys_behavior_sync as ebs


def test_package_imports_and_has_version():
    assert isinstance(ebs.__version__, str)


def test_modules_import():
    from ephys_behavior_sync import events, preprocessing, psth, responsivity, sync, synthetic  # noqa: F401
