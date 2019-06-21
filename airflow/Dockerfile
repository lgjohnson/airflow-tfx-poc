FROM python:3.7.3

#Pipeline Name
ENV PIPELINE_NAME tfx_example_pipeline

COPY ./requirements.txt /tmp/requirements.txt

WORKDIR /airflow_tfx/

# set SLUGIFY_USES_TEXT_UNIDECODE to avoid airflow GPL version: https://www.tensorflow.org/tfx/guide
RUN export SLUGIFY_USES_TEXT_UNIDECODE=yes && \
    pip install -r /tmp/requirements.txt

# create directories for pipeline code

ENV DAGS_DIR "${AIRFLOW_HOME}/dags"
ENV DATA_DIR "${AIRFLOW_HOME}/data/${PIPELINE_NAME}"
ENV PLUGINS_DIR = "${AIRFLOW_HOME}/plugins/${PIPELINE_NAME}"
ENV SERVING_DIR = "${AIRFLOW_HOME}/models/${PIPELINE_NAME}"

RUN mkdir -p ${DAGS_DIR} ${DATA_DIR} ${PLUGINS_DIR}

# copy pipeline code into airflow dags folder
COPY ./pipeline.py $DAGS_DIR

# copy example csv into data directory
COPY ./example.csv $DATA_DIR

ENTRYPOINT airflow webserver -p 8080 & airflow scheduler
