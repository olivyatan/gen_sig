FROM hub.tess.io/krylov_curated_workspace/base:cuda12.2-devel-py3.10-ubuntu20.04.latest

MAINTAINER Gilad Wagner "gwagner@ebay.com"

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y less
RUN pip install --upgrade pip
RUN apt-get install graphviz -y

COPY dist/generative_signals-0.1.0-py3-none-any.whl /tmp/
RUN pip install --index-url https://artifactory.corp.ebay.com/artifactory/api/pypi/PyPI/simple \
    --extra-index-url https://artifactory.corp.ebay.com/artifactory/api/pypi/pypi-bx/simple \
    '/tmp/generative_signals-0.1.0-py3-none-any.whl[krylov]'
