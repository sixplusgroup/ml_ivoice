FROM pre_env:latest

ADD . /code

ENV PYTHONPATH "${PYTHONPATH}:/home/doomsdor/ml_ivoice"

WORKDIR /code

RUN pip install -r requirements.txt

CMD python /code/grpc/ivoice_server.py

EXPOSE 9000
