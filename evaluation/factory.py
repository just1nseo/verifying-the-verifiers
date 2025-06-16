from common import shared_config
from evaluation.tasks import FactVerificationDataFormat, FactVerificationLabel
from evaluation.tasks.dataset import (load_clearfacts, load_grayfacts)
from models import OpenAIModel, GeminiModel, AnthropicModel
from models.attributor import (
    APIFactAttributorFewShot, 
    APIFactAttributorZeroShot, 
    OpenFactAttributorFewShot, 
    OpenFactAttributorZeroShot, 
    ClearCheckDirect,
    ClearCheckCoT,
    FactAttributorBase
)


def load_fact_verification_dataset(
    dataset_name: str,
) -> list[FactVerificationDataFormat]:
    """Load supported datasets: clearfacts and grayfacts"""
    if dataset_name == "clearfacts":
        return load_clearfacts()
    elif dataset_name == "grayfacts":
        return load_grayfacts()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: clearfacts, grayfacts")


def load_model(model_type: str, model_name: str, num_processes: int = 8) -> FactAttributorBase:
    """Load supported attributor models from models.attributor"""
    
    if model_type == "APIFactAttributorFewShot":
        # Check if this is a Gemini model
        if "GEMINI:" in shared_config.model_options[model_name]:
            rater_model = GeminiModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        elif "ANTHROPIC:" in shared_config.model_options[model_name]:
            rater_model = AnthropicModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            rater_model = OpenAIModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        return APIFactAttributorFewShot(rater_model, num_processes=num_processes)
    
    elif model_type == "APIFactAttributorZeroShot":
        if "GEMINI:" in shared_config.model_options[model_name]:
            rater_model = GeminiModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        elif "ANTHROPIC:" in shared_config.model_options[model_name]:
            rater_model = AnthropicModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            rater_model = OpenAIModel(
                shared_config.model_options[model_name],
                temperature=0.7,
                max_tokens=1024,
            )
        return APIFactAttributorZeroShot(rater_model, num_processes=num_processes)
    
    elif model_type == "OpenFactAttributorFewShot":
        return OpenFactAttributorFewShot(
            model_path=model_name,
            tensor_parallel_size=num_processes,
        )
    
    elif model_type == "OpenFactAttributorZeroShot":
        return OpenFactAttributorZeroShot(
            model_path=model_name,
            tensor_parallel_size=num_processes,
        )
    
    elif model_type == "ClearCheckDirect":
        return ClearCheckDirect(model_name)
    
    elif model_type == "ClearCheckCoT":
        return ClearCheckCoT(model_name)
    
    else:
        supported_models = [
            "APIFactAttributorFewShot",
            "APIFactAttributorZeroShot", 
            "OpenFactAttributorFewShot",
            "OpenFactAttributorZeroShot",
            "OpenFactAttributor",
            "OpenFactAttributorCoT"
        ]
        raise ValueError(f"Unknown model: {model_type}. Supported models: {supported_models}")
