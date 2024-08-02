
from typing import NamedTuple, Dict, Text, Any

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras import layers
from keras_tuner.engine import base_tuner
from transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

# Define a NamedTuple to hold the results of the tuning function
TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

# Early stopping callback to avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=10
)

def gzip_reader_fn(filenames):
    """Read TFRecord files with GZIP compression.

    Args:
        filenames (str): Path to the TFRecord file(s).

    Returns:
        tf.data.TFRecordDataset: Dataset with GZIP-compressed data.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Create a dataset for training or evaluation.

    Args:
        file_pattern (str): Pattern for input TFRecord files.
        tf_transform_output (tft.TFTransformOutput): Transform output object.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        tf.data.Dataset: Dataset containing (features, labels) tuples.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

def get_tuner_model(hyperparameters, show_summary=True):
    """Define a Keras model for hyperparameter tuning.

    Args:
        hyperparameters (kt.HyperParameters): Hyperparameters for tuning.
        show_summary (bool, optional): Whether to print model summary. Defaults to True.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    num_layers = hyperparameters.Choice('num_layers', values=[1, 2, 3])
    dense_units = hyperparameters.Int('dense_units', min_value=16, max_value=256, step=16)
    dropout_rate = hyperparameters.Float('dropout_rate', min_value=0.1, max_value=0.7, step=0.1)
    learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            layers.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            layers.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = layers.concatenate(input_features)
    deep = layers.Dense(dense_units, activation='relu')(concatenate)

    for _ in range(num_layers):
        deep = layers.Dense(dense_units, activation='relu')(deep)

    deep = layers.Dropout(dropout_rate)(deep)
    outputs = layers.Dense(1, activation='sigmoid')(deep)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['binary_accuracy']
    )

    if show_summary:
        model.summary()

    return model

def tuner_fn(fn_args):
    """Tune the model to find the best hyperparameters.

    Args:
        fn_args (NamedTuple): Contains information for tuning.

    Returns:
        TunerFnResult: Contains the tuner and fit arguments.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files[0], tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files[0], tf_transform_output)

    tuner = kt.Hyperband(
        hypermodel=get_tuner_model,
        objective=kt.Objective('val_loss', direction='min'),
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name='stroke_disaster_kt'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [early_stop]
        }
    )
