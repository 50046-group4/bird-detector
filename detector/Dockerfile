FROM python:3.8

RUN apt update && \
	apt-get install curl gnupg ca-certificates zlib1g-dev libjpeg-dev git -y

COPY requirements.txt /app/
RUN pip3 install --extra-index-url https://google-coral.github.io/py-repo/ --requirement /app/requirements.txt
COPY . /app/

CMD /bin/sh