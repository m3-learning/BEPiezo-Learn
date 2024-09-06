"""
    Dummy conftest.py for bepiezo_learn.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest

import pytest

# You can add more notebooks to this list if needed
notebooks_to_test = ['1(optional)_Noisy_Data_and_Fitting.ipynb']

@pytest.mark.parametrize("notebook", notebooks_to_test)
def test_notebook_execution(notebook):
    # Run pytest with nbval on the notebook
    pytest.main([notebook, "--nbval-lax"])
