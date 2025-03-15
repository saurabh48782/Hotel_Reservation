# Hotel Reservation Cancellation Prediction
--------------------------
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediction of cancellations of reservations at a Hotel.

## One-time development setup
Install `pyenv`.

```
curl https://pyenv.run | bash
```
and then run the following to add to your `.bashrc` (for `.zshrc` see the pyenv documentation).
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```
Now install the packages.
```
pip install poetry
poetry install
```
Install the pre-commit hook by running
```
poetry run pre-commit install
```
