FROM pytorch/pytorch

RUN pip install ipykernel && \
        python -m ipykernel install --sys-prefix

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /
COPY ./data data
COPY ./src src
COPY pipeline.py pipeline.py
RUN mkdir output

EXPOSE 8080
