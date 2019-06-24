# POC for Airflow-backed Tensorflow Extended Pipeline

* Data `airflow/iris.csv` are the iris dataset, without the column of flower names. 150 rows by 4 columns (all features are numeric).

To run in dev (and see code changes apply immediately): `docker-compose up -f docker-composer.dev.yaml -d`.
To run the actual service `docker-compose  up  -d`.
