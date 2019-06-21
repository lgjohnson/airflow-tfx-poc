FROM python:3.7.3

COPY ./requirements.txt /tmp/requirements.txt

WORKDIR /airflow_tfx/

# set SLUGIFY_USES_TEXT_UNIDECODE to avoid GPL version: https://www.tensorflow.org/tfx/guide
RUN export SLUGIFY_USES_TEXT_UNIDECODE=yes && \
    pip install -r /tmp/requirements.txt

# create directories for pipeline code
RUN mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/data $AIRFLOW_HOME/plugins

COPY 
