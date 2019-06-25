# POC for Airflow-backed Tensorflow Extended Pipeline

* Data `airflow/iris.csv` are the iris dataset, modified so the flower names is a binary output indicating whether the sample is 'Iris Setosa'. 150 rows by 5 columns (all features are numeric; label `class` is binary.


To run in dev (and see code changes apply immediately): `docker-compose up -f docker-composer.dev.yaml -d`.
To run the actual service `docker-compose  up  -d`.
