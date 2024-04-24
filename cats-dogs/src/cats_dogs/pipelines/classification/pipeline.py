"""
This is a boilerplate pipeline 'classification'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_images, data_split, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_images,
            inputs="cat_dog",
            outputs="preprocessed",
            name="preprocess_images",
        ),
        node(
            func=data_split,
            inputs="preprocessed",
            outputs=["x_train", "x_test", "y_train", "y_test"],
            name="data_split",
        ),
        node(
            func=train_model,
            inputs=["x_train", "y_train"],
            outputs="model",
            name="train_model",
        ),
        node(
            func=evaluate_model,
            inputs=["model", "x_test", "y_test"],
            outputs="evaluation",
            name="evaluate_model",
        ),
    ])
