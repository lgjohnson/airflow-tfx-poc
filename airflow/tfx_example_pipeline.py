import os
import datetime
import logging
from airflow import PipelineDecorator

from tfx import logging_utils
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.tfx_runner import TfxRunner
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
SERVING_DIR = os.environ['SERVING_DIR']     # directory to output models

# Model Code Files
TRANSFORM_MODULE_FILE = os.environ['TRANSFORM_MODULE_FILE']  # transform code
MODEL_MODULE_FILE = os.environ['MODEL_MODULE_FILE']          # model train code


@PipelineDecorator(
    pipeline_name=PIPELINE_NAME,
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=datetime.datetime(2018, 1, 1),
    enable_cache=True,
    additional_pipeline_args={
        'logger_args': logging_utils.LoggerConfig(
            log_root='/var/tmp/tfx/logs', 
            log_level=logging.INFO
        )
    },
    metadata_db_root=os.path.join(HOME_DIR, 'data/tfx/metadata'),
    pipeline_root=DAGS_DIR
)
def create_pipeline():

    # Read data in; can split data here
    examples = csv_input(DATA_DIR)
    example_gen = CsvExampleGen(input_data=examples)

    # Generate feature statistics
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.output)

    # Infer schema for data
    infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

    # Identify  anomomalies in training and serving data
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output,
        schema=infer_schema.outputs.output
    )

    # Performs feature engineering; emits a SavedModel that does preprocessing
    transform = Transform(
        input_data=example_gen.outputs.output,
        schema=infer_schema.outputs.output,
        module_file=TRANSFORM_MODULE_FILE
    )

    # Trains a model
    trainer = Trainer(
        module_file=MODEL_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_steps=10000,
        eval_steps=5000,
        warm_starting=True
    )

    # Evaluates the model on  different slices of the data (bias detection?!)
    model_analyzer = Evaluator(
        examples=example_gen.outputs.output,
        model_exports=trainer.outputs.output
    )

    # Compares new model against a baseline; both models evaluated on a dataset
    model_validator = ModelValidator(
        examples=example_gen.outputs.output,
        model=trainer.outputs.output
    )

    # Pushes a blessed model to a deployment target (tfserving)
    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        serving_model_dir=SERVING_DIR
    )

    return [
        example_gen, statistics_gen, infer_schema, validate_stats, transform,
        trainer, model_analyzer, model_validator, pusher
    ]

pipeline = TfxRunner().run(create_pipeline())
