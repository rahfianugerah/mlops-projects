"""
Initiate tfx pipeline components
"""
 
import os
 
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, 
    StatisticsGen, 
    SchemaGen, 
    ExampleValidator, 
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2 
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)
 
def init_components(
    data_dir,
    transform_module,
    tuner_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    """Initiate tfx pipeline components
 
    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory
 
    Returns:
        TFX components
    """
    # Define the output configuration for the CsvExampleGen component
    # This configuration specifies how the input data should be split into training and evaluation sets
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                # Define the training split with 8 hash buckets
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                
                # Define the evaluation split with 2 hash buckets
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
            ]
        )
    )

    # Initialize the CsvExampleGen component with the input data directory and output configuration
    example_gen = CsvExampleGen(input_base=data_dir, output_config=output)
    
    # Initialize the StatisticsGen component
    # This component generates statistics for the dataset, which helps in understanding the data distribution
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Initialize the SchemaGen component
    # This component generates a schema based on the dataset statistics, which defines the expected data types and constraints for each feature
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )
    
    # Initialize the ExampleValidator component
    # This component validates the dataset against the generated schema and statistics to identify data quality issues
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # Initialize the Transform component
    # This component applies data transformations to the dataset according to the specified schema and transformation logic
    transform = Transform(
        examples=example_gen.outputs["examples"],  # The examples output from the CsvExampleGen component
        schema=schema_gen.outputs["schema"],       # The schema output from the SchemaGen component
        module_file=os.path.abspath(transform_module)  # The path to the module file containing the transformation logic
    )

    # Initialize the Tuner component
    # This component performs hyperparameter tuning using the specified tuning logic and dataset
    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),  # The path to the module file containing the tuning logic
        examples=transform.outputs['transformed_examples'],  # The transformed dataset output from the Transform component
        transform_graph=transform.outputs['transform_graph'],  # The transformation graph output from the Transform component
        schema=schema_gen.outputs['schema'],  # The schema output from the SchemaGen component
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=training_steps),  # Training arguments specifying the training split and number of steps
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=eval_steps)  # Evaluation arguments specifying the evaluation split and number of steps
    )

    # Initialize the Trainer component
    # This component trains a model using the specified training logic, transformed data, and hyperparameters
    trainer = Trainer(
        module_file=os.path.abspath(training_module),  # The path to the module file containing the training logic
        examples=transform.outputs['transformed_examples'],  # The transformed dataset output from the Transform component
        transform_graph=transform.outputs['transform_graph'],  # The transformation graph output from the Transform component
        schema=schema_gen.outputs['schema'],  # The schema output from the SchemaGen component
        hyperparameters=tuner.outputs['best_hyperparameters'],  # The best hyperparameters identified by the Tuner component
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=training_steps),  # Training arguments specifying the training split and number of steps
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=eval_steps)  # Evaluation arguments specifying the evaluation split and number of steps
    )
    
    # Initialize the Resolver component
    # This component resolves the latest blessed model based on the provided strategy and channels
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,  # Strategy for selecting the latest blessed model
        model=Channel(type=Model),  # Channel providing model artifacts
        model_blessing=Channel(type=ModelBlessing)  # Channel providing model blessing artifacts
    ).with_id("Latest_blessed_model_resolver")  # Assign an ID to the resolver component

    
    # Define slicing specifications for evaluation
    # This specifies how the data should be sliced for evaluation, allowing metrics to be computed for different slices of the data
    slicing_specs = [
        tfma.SlicingSpec(),  # Default slicing (overall metrics)
        tfma.SlicingSpec(feature_keys=['gender', 'ever_married'])  # Metrics sliced by 'gender' and 'ever_married' features
    ]

    # Define metrics specifications for evaluation
    # This specifies which metrics to compute and their configurations during evaluation
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),  # Area Under the Curve
            tfma.MetricConfig(class_name='ExampleCount'),  # Count of examples
            tfma.MetricConfig(class_name='TruePositives'),  # Count of true positives
            tfma.MetricConfig(class_name='FalsePositives'),  # Count of false positives
            tfma.MetricConfig(class_name='TrueNegatives'),  # Count of true negatives
            tfma.MetricConfig(class_name='FalseNegatives'),  # Count of false negatives
            tfma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.5}  # Threshold for the metric
                    ),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.0001}  # Change threshold for the metric
                    )
                )
            )
        ])
    ]
 
    # Define the evaluation configuration
    # This specifies the model specifications, slicing specifications, and metrics specifications to be used for model evaluation
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='stroke')],  # Model specification with the label key 'stroke'
        slicing_specs=slicing_specs,  # Define how data should be sliced for evaluation
        metrics_specs=metrics_specs  # Define which metrics to compute and their configurations
    )
    
    # Initialize the Evaluator component
    # This component evaluates the performance of a trained model against a baseline model using the specified evaluation configuration
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],  # The raw dataset output from the CsvExampleGen component
        model=trainer.outputs["model"],  # The trained model output from the Trainer component
        baseline_model=model_resolver.outputs["model"],  # The baseline (latest blessed) model output from the Resolver component
        eval_config=eval_config,  # The evaluation configuration specifying metrics and slicing specs
    )
    
    # Initialize the Pusher component
    # This component is responsible for pushing the trained model to the specified destination for serving
    pusher = Pusher(
        model=trainer.outputs['model'],  # The trained model output from the Trainer component
        model_blessing=evaluator.outputs['blessing'],  # The model blessing output from the Evaluator component, indicating that the model is ready for deployment
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir  # The directory where the model will be pushed for serving
            )
        )
    )
    
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
    
    return components