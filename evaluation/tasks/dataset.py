from evaluation.tasks import FactVerificationDataFormat, FactVerificationLabel
import json

def load_clearfacts() -> list[FactVerificationDataFormat]:
    """
    Load the final judge dataset from the saved jsonl file.
    """
    with open("data/clearfacts.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    return [FactVerificationDataFormat(
        topic=item["topic"],
        statement=item["statement"],
        reference_documents=item["reference_documents"],
        label=(FactVerificationLabel.SUPPORTED if item["label"] == "S" 
               else FactVerificationLabel.AMBIGUOUS if item["label"] == "AMBIG"
               else FactVerificationLabel.NOT_SUPPORTED if item["label"] == "NS"
               else FactVerificationLabel(item["label"])),  # Fallback for other label values
        category=item["category"],
        additional_info=item["additional_info"]
    ) for item in data]

def load_grayfacts() -> list[FactVerificationDataFormat]:
    """
    Load the final judge dataset from the saved jsonl file.
    """
    with open("data/grayfacts.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    return [FactVerificationDataFormat(
        topic=item["topic"],
        statement=item["statement"],
        reference_documents=item["reference_documents"],
        label=(FactVerificationLabel.SUPPORTED if item["label"] == "S" 
               else FactVerificationLabel.AMBIGUOUS if item["label"] == "AMBIG"
               else FactVerificationLabel.NOT_SUPPORTED if item["label"] == "NS"
               else FactVerificationLabel(item["label"])),  # Fallback for other label values
        category=item["category"],
        additional_info=item["additional_info"]
    ) for item in data]