
import tensorflow as tf
import tensorflow_transform as tft

# Dictionary defining categorical features and their number of categories
CATEGORICAL_FEATURES = {
    'gender': 2,
    'ever_married': 2,
    'work_type': 5,
    'Residence_type': 2,
    'smoking_status': 4
}

# List of numerical features
NUMERICAL_FEATURES = [
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level',
    'bmi'
]

# Key for the label feature
LABEL_KEY = 'stroke'

def transformed_name(key):
    """
    Generate a transformed feature name by appending '_xf' to the original key.

    Args:
        key (str): The original feature key.

    Returns:
        str: Transformed feature key.
    """
    return f"{key}_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label tensor (0 or 1) into a one-hot encoded tensor.

    Args:
        label_tensor (tf.Tensor): Tensor containing the label (0 or 1).
        num_labels (int, optional): Number of labels for one-hot encoding. Defaults to 2.

    Returns:
        tf.Tensor: One-hot encoded tensor with shape [batch_size, num_labels].
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features using TensorFlow Transform.

    Args:
        inputs (dict): Dictionary mapping feature keys to raw feature tensors.

    Returns:
        dict: Dictionary mapping feature keys to transformed feature tensors.
    """
    outputs = {}

    # Process categorical features
    for key, dim in CATEGORICAL_FEATURES.items():
        # Compute and apply vocabulary for categorical features
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        # Convert integer values to one-hot encoding
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    # Process numerical features
    for feature in NUMERICAL_FEATURES:
        # Scale numerical features to the range [0, 1]
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    # Process label feature
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
