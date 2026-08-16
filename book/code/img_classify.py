import torch
import pandas as pd
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


_model = None
_weights = None


def get_model():
    """Load and cache a memory-efficient pretrained image classifier."""
    global _model, _weights

    if _model is None:
        _weights = MobileNet_V3_Small_Weights.DEFAULT
        _model = mobilenet_v3_small(weights=_weights)
        _model.eval()

    return _model, _weights


def classify_image(image_path, topn=4):
    """Classify an image using pretrained MobileNetV3-Small.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the image.
    topn : int, default=4
        Number of top predictions to return.

    Returns
    -------
    pandas.DataFrame
        Predicted ImageNet classes and probability scores.
    """
    model, weights = get_model()

    image = Image.open(image_path).convert("RGB")

    preprocess = weights.transforms()
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)

    probabilities = torch.softmax(logits[0], dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, topn)

    class_labels = weights.meta["categories"]

    return pd.DataFrame(
        {
            "Class": [class_labels[i] for i in top_indices],
            "Probability score": [
                round(prob.item(), 3) for prob in top_probabilities
            ],
        }
    )
