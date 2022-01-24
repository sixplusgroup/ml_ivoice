FROM pre_env:latest

ADD ./ml_ivoice /code

ENV PYTHONPATH "${PYTHONPATH}:/home/doomsdor/dockerify/ml_ivoice"

WORKDIR /code

RUN pip install -r requirements.txt

CMD python /code/grpc/ivoice_server.py

EXPOSE 9000
