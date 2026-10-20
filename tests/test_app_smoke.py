from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ENTRYPOINT = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_loads_without_exceptions():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=15)

    assert not at.exception, f"App crashed with exceptions: {at.exception}"


def test_switch_language_smoke():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=15)


    # Check that selector exists and change to uz
    if at.selectbox:
        at.selectbox[0].select("uz").run(timeout=15)
        assert not at.exception

        # Switch to en
        at.selectbox[0].select("en").run(timeout=15)
        assert not at.exception
