FROM hub.tess.io/mlplatform/ray-base:ray2.35.0-cuda12.1-py3.10

MAINTAINER Gilad Wagner "gwagner@ebay.com"

RUN apt-get update -y && apt-get upgrade -y
RUN pip install --upgrade pip

COPY dist/generative_signals-0.1.0-py3-none-any.whl /tmp/
RUN pip install --index-url https://artifactory.corp.ebay.com/artifactory/api/pypi/PyPI/simple \
    --extra-index-url https://artifactory.corp.ebay.com/artifactory/api/pypi/pypi-bx/simple \
    '/tmp/generative_signals-0.1.0-py3-none-any.whl[krylov]'
