import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "class"
FEATURE_KEY = "text"
NUM_EPOCHS = 2

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)


def transformed_name(key):
    return f"{key}_xf"


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    
    return dataset

def model_builder(hypa, vectorizer_layer):

  # Define hyperparameters (consider reducing the range if needed)
  hidden_layers = hypa.Choice("hidden_layers", values=[1, 2])
  embedding_size = hypa.Int("embedding_size", min_value=32, max_value=128, step=32)
  lstm_units = hypa.Int("lstm_units", min_value=32, max_value=128, step=32)
  dropout_rate = hypa.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
  dense_units = hypa.Int("dense_units", min_value=32, max_value=128, step=32)  # New hyperparameter

  # Model construction
  inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

  x = vectorizer_layer(inputs)
  x = layers.Embedding(5000, embedding_size)(x)
  x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

  # Add dense layers based on hyperparameter
  for _ in range(hidden_layers):
    x = layers.Dense(dense_units, activation="relu")(x)  # Use dense_units hyperparameter
    x = layers.Dropout(dropout_rate)(x)

  outputs = layers.Dense(1, activation="sigmoid")(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  # Compile for binary classification (consider learning rate tuning)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Consider hyperparameter tuning
      loss="binary_crossentropy",
      metrics=["accuracy"],
  )

  return model


def tuner_fn(fn_args):
  # Load the transform graph
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  # Create training and evaluation datasets with transformed features
  train_set = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
  eval_set = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)

  # Extract text features for vectorization
  vectorizer_dataset = train_set.map(lambda f, l: f[transformed_name(FEATURE_KEY)])

  # Define and adapt the text vectorization layer
  vectorizer_layer = layers.TextVectorization(
      max_tokens=1000, output_mode="int", output_sequence_length=100
  )
  vectorizer_layer.adapt(vectorizer_dataset)

  # Create a Hyperband tuner with the vectorization layer
  tuner = kt.Hyperband(
      hypermodel=lambda hypa: model_builder(hypa, vectorizer_layer),
      objective=kt.Objective("accuracy", direction="max"),
      max_epochs=NUM_EPOCHS,
      factor=3,
      directory=fn_args.working_dir,
      project_name="kt_hyperband",
  )

  # Define fit arguments with early stopping callback
  fit_kwargs = {
      "callbacks": [early_stopping_callback],
      "x": train_set,
      "validation_data": eval_set,
      "steps_per_epoch": fn_args.train_steps,
      "validation_steps": fn_args.eval_steps,
  }

  return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
