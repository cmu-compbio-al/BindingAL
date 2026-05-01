from .mlp import MLPBinder
from .cross_attn import CrossAttentionBinder

MODEL_REGISTRY = {
    "mlp": MLPBinder,
    "cross_attn": CrossAttentionBinder
}

def get_model(model_type: str, **kwargs):
    """
    Returns a model instance based on the specified model type.

    Args:
        model_type (str): The type of model to create. Supported values are "mlp" and "cross_attn".
        **kwargs: Additional keyword arguments to pass to the model constructor
    
    Returns:
        An instance of the specified model type.
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model type: '{model_type}'. Please choose from: [{supported}]")

    return MODEL_REGISTRY[model_type](**kwargs)

__all__ = ["MLPBinder", "CrossAttentionBinder", "get_model", "MODEL_REGISTRY"]
