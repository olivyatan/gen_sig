# generative-signals
Generate Item Signals using LLMs


### Setup
#### 1. Exit All Virtual Environments
#### 2. Install pip globally (if you dan't have it)
```shell
python3 -m ensurepip --upgrade
```
#### 3. Install poetry globally
```shell
pip install poetry==1.5.1 --index-url="https://artifactory.corp.ebay.com/artifactory/api/pypi/PyPI/simple"
poetry install --with test --all-extras
poetry run pre-commit install
```
#### 4. Configure Poetry to use In-Project Virtual Environments
To ensure that Poetry creates virtual environments within the project directory set this configuration:
```shell
poetry config virtualenvs.in-project true
```
TIP: to synchronize the environment after testing changes:
```shell
poetry install --with test --all-extras --sync
```