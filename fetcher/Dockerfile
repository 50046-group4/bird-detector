FROM python:3.7-alpine AS compile-image

RUN apk --no-cache add musl-dev linux-headers g++ python3-dev musl-dev
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /app/
RUN pip3 install --requirement /app/requirements.txt
COPY . /app/

FROM python:3.7-alpine AS build-image
LABEL name=bird-detector-fetcher
# copy the python virtual environment
COPY --from=compile-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

CMD /bin/sh