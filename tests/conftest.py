from _pytest.monkeypatch import MonkeyPatch

from tranquilo.config import IS_OPTIMAGIC_INSTALLED

if IS_OPTIMAGIC_INSTALLED:
    import optimagic.optimizers.tranquilo as t

    _mp = MonkeyPatch()

    def pytest_configure(config):
        _mp.setattr(
            t,
            "IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0",
            "patched-value",
            raising=True,
        )

    def pytest_unconfigure(config):
        _mp.undo()
