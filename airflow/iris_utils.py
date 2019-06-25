#!/usr/bin/env python3

'''
Python source file for Tensorflow transforms
'''

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma


_DENSE_FLOAT_FEATURE_KEYS = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

_LABEL_KEY = 'class'


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _get_raw_feature_spec(schema):
    return tft.tf_metadata.schema_utils\
        .schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    '''Utility that returns a record reader that reads gzip-ed files'''
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP'
    )


def _fill_in_missing(x):
    '''Fills missing  values with '' or 0 and converts to a dense vector'''
    default_value = ''  if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0],  1]),
            default_value
        ),
        axis=1
    )



def preprocessing_fn(inputs):
    '''
    tf.transform's callback fx for preprocessing
    Args:
        inputs: map from feature keys to  raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    '''
    outputs = {}
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(
                inputs[key]
            )
        )

    outputs[_transformed_name(_LABEL_KEY)] = tf.cast(
        _fill_in_missing(
            inputs[_LABEL_KEY]
        ),
        tf.int64
    )

    return outputs


def _build_estimator(config,  hidden_units=None, warm_start_from=None):
    '''
    Build an estimator
    Args:
        config: tf.estimator.RunConfig defines runtime environment
        hidden_units: [int], layer sizes of DNN
        warm_start_from: directory to start from (optional)
    Returns:
        dict of:
            - estimator
            - train_spec
            - eval_spec
            - eval_input_receiver_fn
    '''

    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    return tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        dnn_feature_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25],
        warm_start_from=warm_start_from
    )


def _example_serving_receiver_fn(tf_transform_output, schema):
    '''
    build serving in inputs
    Args:
        tf_transform_output
        schema: schema of input data
    Returns:
        Tensorflow graph which parses examples, applying tf-transform
    '''
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop(_LABEL_KEY)
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec,
        default_batch_size=None
    )
    serving_input_receiver = raw_input_fn()

    transformed_features = tf.transform_output.transform_raw_features(
        serving_input_receiver.features
    )

    return tf.estimator.export.ServingInputReceiver(
        transformed_features,
        serving_input_receiver.receiver_tensors
    )


def _eval_input_receiver_fn(tf_transform_output, schema):
    '''
    build for tf-model-analysis to run model
    Args:
        tf_transform_output
        schema: schema of input data
        Returns:
            EvalInputReceiver function which contains:
                Tensorflow graph that parses raw untransformed features,
                applying tf-transform set of raw, untransformed features
                label against which predictions are compared
    '''

    raw_feature_spec = _get_raw_feature_spec(schema)

    serialized_tf_example = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name='input_example_tensor'
    )

    features =  tf.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = tf_transform_output.transform_raw_features(features)
    receiver_tensors = {'examples': serialized_tf_example}
    features.update(transformed_features)

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_transformed_name(_LABEL_KEY)]
    )


def _input_fn(filenames, tf_transform_output, batch_size=200):
    '''
    generates features and labels for training or eval
    Args:
        filenames: [str] list of CSV files to read data from
        tf_transform_output
        batch_size: int of first dimension size of tensors returned by input_fn
    Returns:
        A (features, indices) tuple where features ia  dict of Tensors,
        indices is a single tensor of label indices
    '''
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()\
        .copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        filenames,
        batch_size,
        transformed_feature_spec,
        reader=_gzip_reader_fn
    )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    return transformed_features, transformed_features.pop(
        _transformed_name(_LABEL_KEY)
    )


def trainer_fn(hparams, schema):
    '''
    build the estimator
    Args:
        hparams: hyperparameters to train the model
        schema: holds schema of training examples
    Returns:
        A dict of the following:
            estimator
            train_spec
            eval_spec
            eval_input_receiver_fn: input fx for  eval
    '''
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    train_batch_size = 10
    eval_batch_size = 10

    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)
    
    train_input_fn = lambda: _input_fn(
        hparams.train_files,
        tf_transform_output,
        batch_size=train_batch_size
    )

    eval_input_fn = lambda: _input_fn(
        hparams.eval_files,
        tf_transform_output,
        batch_size=eval_batch_size
    )

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps
    )

    serving_receiver_fn = lambda: _example_serving_receiver_fn(
        tf_transform_output,
        schema
    )

    exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='iris-eval'
    )

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, 
        keep_checkpoint_max=1
    )
    run_config = run_config.replace(model_dir=hparams.serving_model_dir)

    estimator = _build_estimator(
        hidden_units=[
            max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
            for i in range(num_dnn_layers)
        ],
        config=run_config,
        warm_start_from=hparams.warm_start_from
    )

    receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }
