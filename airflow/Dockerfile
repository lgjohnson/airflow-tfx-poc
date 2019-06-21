FROM python:3.7.3

#Pipeline Name
ENV PIPELINE_NAME tfx_example_pipeline

COPY ./requirements.txt /tmp/requirements.txt

WORKDIR /airflow_tfx/

# set SLUGIFY_USES_TEXT_UNIDECODE to avoid airflow GPL version: https://www.tensorflow.org/tfx/guide
RUN export SLUGIFY_USES_TEXT_UNIDECODE=yes && \
    pip install -r /tmp/requirements.txt

# create directories for pipeline code
RUN mkdir -p "${AIRFLOW_HOME}/dags" \
    "${AIRFLOW_HOME}/data/${PIPELINE_NAME}" \ 
    "${AIRFLOW_HOME}/plugins/${PIPELINE_NAME}"

#pipeline code must live in airflow dags folder
COPY ./pipeline.py "${AIRFLOW_HOME}/dags"

ENTRYPOINT airflow webserver -p 8080 & airflow scheduler
