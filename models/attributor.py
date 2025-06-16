import abc
import json
import os
import re
import multiprocessing as mp
import warnings

from tqdm import tqdm

from common import utils
from data.templates import DOCUMENT_PLACEHOLDER,STATEMENT_PLACEHOLDER, FEW_SHOT_TEMPLATE, ZERO_SHOT_TEMPLATE,  CLEARCHECK_COT, CLEARCHECK_DIRECT
from evaluation.tasks import (FactVerificationDataFormat, FactVerificationLabel)
from models import FactVerificationResult, OpenAIModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from models.merge_lora import maybe_merge_lora_model

class OpenModelBase(abc.ABC):
    @staticmethod
    def build_model(trained_model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95):
        from vllm import LLM

        temp_model_path = maybe_merge_lora_model(trained_model_path)
        if temp_model_path is not None:
            trained_model_path = temp_model_path

        model = LLM(
            model=trained_model_path,
            tokenizer=trained_model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
        )
        if temp_model_path is not None:
            # do rm -rf
            os.system(f"rm -rf {temp_model_path}")
            print(f"Removed temporary directory: {temp_model_path}")

        return model

    @staticmethod
    def build_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer
    
    @property
    def stop_token_ids(self):
        """Get stop token IDs for the model"""
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            return [self.tokenizer.eos_token_id]
        return []
class FactAttributorBase(abc.ABC):
    def __init__(self, num_processes: int = 1):
        self.num_processes = num_processes

    @abc.abstractmethod
    def _fact_attribution(
        self, data: FactVerificationDataFormat
    ) -> tuple[FactVerificationLabel, str]:
        raise NotImplementedError

    @staticmethod
    def _map_attribution_to_verification(attribution_answer: str) -> FactVerificationLabel:
        """Map attribution model outputs to verification labels"""
        attribution_answer = attribution_answer.lower().strip()
        if attribution_answer == "attributable":
            return FactVerificationLabel.SUPPORTED
        elif attribution_answer in ["not attributable", "contradictory"]:
            return FactVerificationLabel.NOT_SUPPORTED
        else:
            return FactVerificationLabel.PARSING_ERROR

    def run(self, dataset: list[FactVerificationDataFormat]) -> FactVerificationResult:
        fact_verification_result = FactVerificationResult()
        if self.num_processes > 1:
            with mp.Pool(self.num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._fact_attribution, dataset),
                        total=len(dataset),
                        desc="Running Fact Attribution...",
                    )
                )
        else:
            results = []
            for data in tqdm(dataset, desc="Running Fact Attribution..."):
                results.append(self._fact_attribution(data))

        for data, (result, response) in zip(dataset, results):
            fact_verification_result.add_result(
                data, prediction=result, response=response
            )

        return fact_verification_result


class APIFactAttributorBase(FactAttributorBase):
    """Base class for API-based fact attributors"""
    
    def __init__(self, rater_model: OpenAIModel, num_processes: int = 4, num_votes: int = 1):
        super().__init__(num_processes)
        self.rater_model = rater_model
        self.num_votes = num_votes
        if num_votes != 1:
            raise NotImplementedError("Multiple votes not implemented.")

    def _get_prompt_template(self) -> str:
        """Override in subclasses to specify template"""
        raise NotImplementedError

    def get_prompt_for_fact_attribution(self, statement: str, passages: list[str]) -> str:
        prompt_format = self._get_prompt_template()
        doc = "\n".join(passages)
        prompt = prompt_format.replace(STATEMENT_PLACEHOLDER, statement)
        prompt = prompt.replace(DOCUMENT_PLACEHOLDER, doc)
        return prompt

    def _fact_attribution(
        self, data: FactVerificationDataFormat
    ) -> tuple[FactVerificationLabel, str]:
        try:
            statement = data.statement
            passages = data.reference_documents
            prompt = self.get_prompt_for_fact_attribution(statement, passages)
            
            response = self.rater_model.generate(prompt).strip()
            answer = utils.extract_last_square_brackets(response)
            answer = re.sub(r"[^\w\s]", "", answer).strip()
            
            # Map attribution answer to verification label
            result = self._map_attribution_to_verification(answer)
            return result, response
        except Exception as e:
            print(f"Error in fact attribution: {e}")
            return FactVerificationLabel.PARSING_ERROR, str(e)


class APIFactAttributorFewShot(APIFactAttributorBase):
    def _get_prompt_template(self) -> str:
        return FEW_SHOT_TEMPLATE


class APIFactAttributorZeroShot(APIFactAttributorBase):
    def _get_prompt_template(self) -> str:
        return ZERO_SHOT_TEMPLATE


class OpenFactAttributorBase(FactAttributorBase, OpenModelBase):
    """Base class for open model fact attributors"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.6,
        gpu_memory_utilization: float = 0.9,
    ):
        super().__init__(num_processes=1)
        self.model = self.build_model(model_path, tensor_parallel_size, gpu_memory_utilization)
        self.tokenizer = self.build_tokenizer(model_path)
        self.temperature = temperature

    def _get_prompt_template(self) -> str:
        """Override in subclasses to specify template"""
        raise NotImplementedError

    def _get_sampling_params(self):
        """Override in subclasses to specify sampling parameters"""
        from vllm import SamplingParams
        
        params = {
            "max_tokens": 32768,
            "temperature": self.temperature,
        }
        
        # Only add stop_token_ids if we have actual tokens
        if self.stop_token_ids:
            params["stop_token_ids"] = self.stop_token_ids
        
        return SamplingParams(**params)

    def get_prompt_for_fact_attribution(self, statement: str, passages: list[str]) -> str:
        prompt_format = self._get_prompt_template()
        doc = "\n".join(passages)
        prompt = prompt_format.replace(STATEMENT_PLACEHOLDER, statement)
        prompt = prompt.replace(DOCUMENT_PLACEHOLDER, doc)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response - override for different extraction methods"""
        answer = utils.extract_last_square_brackets(response)
        return re.sub(r"[^\w\s]", "", answer).strip()

    def _fact_attribution(
        self, data: FactVerificationDataFormat
    ) -> tuple[FactVerificationLabel, str]:
        try:
            statement = data.statement
            passages = data.reference_documents
            prompt = self.get_prompt_for_fact_attribution(statement, passages)
            sampling_params = self._get_sampling_params()
            
            response = self.model.generate(
                [prompt], sampling_params=sampling_params, use_tqdm=False
            )
            response = response[0].outputs[0].text.strip()
            
            # Extract and map attribution answer to verification label
            answer = self._extract_answer(response)
            result = self._map_attribution_to_verification(answer)
            return result, response
        except Exception as e:
            print(f"Error in fact attribution: {e}")
            return FactVerificationLabel.PARSING_ERROR, str(e)


class OpenFactAttributorFewShot(OpenFactAttributorBase):
    def _get_prompt_template(self) -> str:
        return FEW_SHOT_TEMPLATE


class OpenFactAttributorZeroShot(OpenFactAttributorBase):
    def _get_prompt_template(self) -> str:
        return ZERO_SHOT_TEMPLATE


class ClearCheckDirect(OpenFactAttributorBase):
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1):
        super().__init__(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            temperature=0.0,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
    def _get_prompt_template(self) -> str:
        return CLEARCHECK_DIRECT

    def _get_sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )

    def _extract_answer(self, response: str) -> str:
        """For direct attribution, use the response as-is"""
        return response.strip()


class ClearCheckCoT(OpenFactAttributorBase):
    def __init__(
        self,
        model_path: str,
        num_votes: int = 1,
        temperature: float = 0.1,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.num_votes = num_votes
        if num_votes != 1:
            raise NotImplementedError("Multiple votes not implemented.")

    def _get_prompt_template(self) -> str:
        return CLEARCHECK_COT

    def _get_sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=1024,
            temperature=self.temperature,
            top_p=1.0,
            logprobs=3,
        )