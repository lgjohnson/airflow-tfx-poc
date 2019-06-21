FROM python:3.7.3

#Pipeline Name
ENV PIPELINE_NAME tfx_example_pipeline

# Install Python packages inc. Airflow
# set SLUGIFY_USES_TEXT_UNIDECODE to avoid airflow GPL version: https://www.tensorflow.org/tfx/guide
ENV AIRFLOW_HOME "/airflow_tfx"
COPY ./requirements.txt /tmp/requirements.txt
RUN export SLUGIFY_USES_TEXT_UNIDECODE=yes && \
    pip install -r /tmp/requirements.txt

# create directories for pipeline code
ENV DAGS_DIR "${AIRFLOW_HOME}/dags"
ENV DATA_DIR "${AIRFLOW_HOME}/data/${PIPELINE_NAME}/simple"
ENV METADATA_DIR "${AIRFLOW_HOME}/data/${PIPELINE_NAME}/metadata"
ENV PLUGINS_DIR "${AIRFLOW_HOME}/plugins/${PIPELINE_NAME}"
ENV LOGS_DIR "/var/tmp/tfx/logs"
ENV SERVING_DIR "${AIRFLOW_HOME}/models/${PIPELINE_NAME}"

ENV MODULES_DIR "${AIRFLOW_HOME}/tfx/modules"
ENV TRANSFORM_MODULE_FILE "${MODULES_DIR}/transform.py"
ENV MODEL_MODULE_FILE "${MODULES_DIR}/model.py"

RUN mkdir -p $DAGS_DIR \
    $DATA_DIR \
    $METADATA_DIR \
    $PLUGINS_DIR \
    $LOGS_DIR \
    $SERVING_DIR \
    $MODULES_DIR

# copy pipeline code into airflow dags folder
COPY ./pipeline.py $DAGS_DIR

# copy model code into modules directory
COPY ./transform.py ./model.py $MODULES_DIR/

# copy example csv into data directory
COPY ./example.csv $DATA_DIR

WORKDIR /airflow_tfx/

EXPOSE 8080

# run airflow webserver in background and start scheduler
ENTRYPOINT airflow initdb && \
    airflow webserver -p 8080 & \
    sleep 5s && \
    airflow scheduler
