import dataclasses
import enum


class FactVerificationLabel(enum.Enum):
    SUPPORTED = 'S'
    NOT_SUPPORTED = 'NS'
    AMBIGUOUS = 'AMBIG'
    PARSING_ERROR = 'PARSING_ERROR'


@dataclasses.dataclass
class FactVerificationDataFormat:
    topic: str
    statement: str
    reference_documents: list[str]  # this can be a single document or multiple documents
    label: FactVerificationLabel
    category: str  # this can be task dependent.
    additional_info: dict = dataclasses.field(default_factory=dict)