import os
import datetime
import logging
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.utils.dsl_utils import csv_input

# Airflow config parameters
PIPELINE_NAME = os.environ['PIPELINE_NAME']
SCHEDULE_INTERVAL = os.environ.get('SCHEDULE_INTERVAL', None)

# Directory variables
HOME_DIR = os.environ['AIRFLOW_HOME']       # airflow home directory
DAGS_DIR = os.environ['DAGS_DIR']           # directory for pipeline code
DATA_DIR = os.environ['DATA_DIR']           # directory for input data
METADATA_DIR = os.environ['METADATA_DIR']   # directory for metadata
PLUGINS_DIR = os.environ['PLUGINS_DIR']     # directory for plugins
LOGS_DIR = os.environ['LOGS_DIR']           # directory for logs
SERVING_DIR = os.environ['SERVING_DIR']     # directory to output models

# Model Code Files
TRANSFORM_MODULE_FILE = os.environ['TRANSFORM_MODULE_FILE']  # transform code
MODEL_MODULE_FILE = os.environ['MODEL_MODULE_FILE']          # model train code

AIRFLOW_CONFIG = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2018, 1, 1)
}


def create_pipeline():

    # Read data in; can split data here
    examples = csv_input(DATA_DIR)
    example_gen = CsvExampleGen(input_base=examples, name='iris_example')

    # Generate feature statistics
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

    # Infer schema for data
    infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

    # Identify  anomomalies in training and serving data
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output,
        schema=infer_schema.outputs.output
    )

    # Performs feature engineering; emits a SavedModel that does preprocessing
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=TRANSFORM_MODULE_FILE
    )

    # Trains a model
    trainer = Trainer(
        module_file=MODEL_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )

    # Evaluates the model on  different slices of the data (bias detection?!)
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output
    )

    # Compares new model against a baseline; both models evaluated on a dataset
    model_validator = ModelValidator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.output
    )

    # Pushes a blessed model to a deployment target (tfserving)
    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_DIR
            )
        )
    )

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=DAGS_DIR,
        components=[
            example_gen,
            statistics_gen,
            infer_schema,
            validate_stats,
            transform,
            trainer,
            model_analyzer,
            model_validator,
            pusher
        ],
        enable_cache=True,
        metadata_db_root=METADATA_DIR,
        additional_pipeline_args={
            'logger_args': {
                'log_root': LOGS_DIR,
                'log_level': logging.INFO
            }
        }
    )


airflow_pipeline = AirflowDAGRunner(AIRFLOW_CONFIG).run(create_pipeline())
