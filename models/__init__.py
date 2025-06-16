import abc
import json
import multiprocessing as mp
import os
import re
import warnings

import numpy as np
import rank_bm25
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

from common import utils
from evaluation.tasks import FactVerificationDataFormat, FactVerificationLabel
from models.openai_utils import (get_avior_chat_response_sync,
                                 get_openai_chat_response_sync)
from models.gemini_utils import get_gemini_chat_response_sync
from models.anthropic_utils import get_anthropic_chat_response_sync


def get_answer(answer: str) -> str:
    answer = utils.extract_last_square_brackets(answer)
    answer = re.sub(r"[^\w\s]", "", answer).strip()
    return answer


class FactVerificationResult:
    def __init__(self):
        self.results = []

    def add_result(
        self,
        data: FactVerificationDataFormat,
        prediction: FactVerificationLabel,
        response: str,
    ):
        self.results.append(
            {
                "topic": data.topic,
                "statement": data.statement,
                "reference_documents": data.reference_documents,
                "label": data.label,
                "category": data.category,
                "prediction": prediction,
                "response": response,
                "additional_info": data.additional_info,
            }
        )

    def _get_dataset_type(self) -> str:
        """Determine dataset type based on labels present"""
        labels = set(result["label"] for result in self.results)
        if FactVerificationLabel.AMBIGUOUS in labels:
            return "grayfacts"
        else:
            return "clearfacts"

    def _report_clearfacts(self, results: list[dict], print_result: bool = True) -> dict:
        """Generate metrics for clearfacts"""
        # get F1, precision, recall, macro F1, balanced accuracy.
        num_results = len(results)
        ground_truth_labels = [
            result["label"] == FactVerificationLabel.SUPPORTED for result in results
        ]
        predicted_labels = [
            result["prediction"] == FactVerificationLabel.SUPPORTED
            for result in results
        ]
        num_parsing_errors = len(
            [
                result
                for result in results
                if result["prediction"] == FactVerificationLabel.PARSING_ERROR
            ]
        )
        num_gt_supported = sum(ground_truth_labels)
        num_gt_not_supported = len(ground_truth_labels) - num_gt_supported
        num_predicted_supported = sum(predicted_labels)
        num_predicted_not_supported = len(predicted_labels) - num_predicted_supported

        # get response stats.
        responses = [result["response"] for result in results]
        answers = [get_answer(response) for response in responses]
        num_pred_attributable = sum([answer.strip().lower() == "attributable" for answer in answers])
        num_pred_not_attributable = sum([answer.strip().lower() == "not attributable" for answer in answers])
        num_pred_contradictory = sum([answer.strip().lower() == "contradictory" for answer in answers])

        # don't show warning for zero division.
        warnings.filterwarnings("ignore")
        precision = precision_score(
            ground_truth_labels, predicted_labels, zero_division=0
        )
        recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
        false_recall = recall_score(
            ground_truth_labels, predicted_labels, pos_label=False, zero_division=0
        )
        f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)
        macro_f1 = f1_score(
            ground_truth_labels, predicted_labels, average="macro", zero_division=0
        )
        balanced_accuracy = balanced_accuracy_score(
            ground_truth_labels,
            predicted_labels,
        )

        warnings.filterwarnings("default")
        if print_result:
            utils.print_color(f"Dataset Type: Clearfacts", "blue")
            utils.print_color(f"Number of results: {num_results}", "green")
            utils.print_color(f"Number of supported: {num_gt_supported}", "green")
            utils.print_color(
                f"Number of not supported: {num_gt_not_supported}", "green"
            )
            utils.print_color(
                f"Number of predicted supported: {num_predicted_supported}", "green"
            )
            utils.print_color(
                f"Number of predicted not supported: {num_predicted_not_supported}",
                "green",
            )
            utils.print_color(
                f"Number of parsing errors: {num_parsing_errors}", "green"
            )
            
            # Attribution breakdown
            utils.print_color("\nModel Output Breakdown:", "yellow")
            utils.print_color(
                f"Predicted 'Attributable': {num_pred_attributable}", "green"
            )
            utils.print_color(
                f"Predicted 'Not Attributable': {num_pred_not_attributable}", "green"
            )
            utils.print_color(
                f"Predicted 'Contradictory': {num_pred_contradictory}", "green"
            )
            
            # Key metrics for clearfacts
            utils.print_color("\nKey Metrics for Clearfacts:", "yellow")
            utils.print_color(f"Macro F1: {macro_f1:.3f}", "green")
            utils.print_color(f"Balanced Accuracy: {balanced_accuracy:.3f}", "green")
            utils.print_color(f"Precision: {precision:.3f}", "green")
            utils.print_color(f"Recall: {recall:.3f}", "green")
            utils.print_color(f"F1: {f1:.3f}", "green")
            
        return {
            "dataset_type": "clearfacts",
            "num_results": num_results,
            "num_gt_supported": num_gt_supported,
            "num_gt_not_supported": num_gt_not_supported,
            "num_parsing_errors": num_parsing_errors,
            "num_predicted_supported": num_predicted_supported,
            "num_predicted_not_supported": num_predicted_not_supported,
            "num_pred_attributable": num_pred_attributable,
            "num_pred_not_attributable": num_pred_not_attributable,
            "num_pred_contradictory": num_pred_contradictory,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "false_recall": false_recall,
        }

    def _report_grayfacts(self, results: list[dict], print_result: bool = True) -> dict:
        """Generate metrics for Grayfacts"""
        # Split results by label type
        normal_results = [r for r in results if r["label"] in [FactVerificationLabel.SUPPORTED, FactVerificationLabel.NOT_SUPPORTED]]
        ambiguous_results = [r for r in results if r["label"] == FactVerificationLabel.AMBIGUOUS]
        
        num_results = len(results)
        num_normal = len(normal_results)
        num_ambiguous = len(ambiguous_results)
        total_parsing_errors = sum(1 for r in results if r["prediction"] == FactVerificationLabel.PARSING_ERROR)
        
        # Initialize metrics
        macro_f1 = 0.0
        balanced_accuracy = 0.0
        s_pct = 0.0
        original_macro_f1 = 0.0
        
        if print_result:
            utils.print_color(f"Dataset Type: Grayfacts", "blue")
            utils.print_color(f"Total examples: {num_results}", "green")
            utils.print_color(f"Total parsing errors: {total_parsing_errors} ({total_parsing_errors/num_results*100:.1f}%)", "green")
        
        # Check if we have original labels and calculate original macro F1
        results_with_original_labels = [r for r in results if r.get("additional_info", {}).get("original_label")]
        original_metrics = {}
        
        if results_with_original_labels:
            # Create pseudo-results with original labels for evaluation
            original_eval_results = []
            for r in results_with_original_labels:
                original_label = r["additional_info"]["original_label"]
                if original_label in ["S", "NS"]:
                    # Convert to FactVerificationLabel format
                    original_label_enum = FactVerificationLabel.SUPPORTED if original_label == "S" else FactVerificationLabel.NOT_SUPPORTED
                    pseudo_result = {
                        **r,
                        "label": original_label_enum
                    }
                    original_eval_results.append(pseudo_result)
            
            if original_eval_results:
                original_metrics = self._report_clearfacts(original_eval_results, print_result=False)
                original_macro_f1 = original_metrics["macro_f1"]
                
                original_s_count = sum(1 for r in results_with_original_labels if r.get("additional_info", {}).get("original_label") == "S")
                original_ns_count = sum(1 for r in results_with_original_labels if r.get("additional_info", {}).get("original_label") == "NS")
                
                if print_result:
                    utils.print_color("\n--- Original Labels Metrics ---", "yellow")
                    utils.print_color(f"Original Supported (S): {original_s_count}", "green")
                    utils.print_color(f"Original Not Supported (NS): {original_ns_count}", "green")
                    utils.print_color(f"Macro F1 of Original Labels: {original_macro_f1:.3f}", "green")
        
        # Evaluate normal examples using standard metrics
        normal_metrics = {}
        if normal_results:
            normal_metrics = self._report_clearfacts(normal_results, print_result=False)
            macro_f1 = normal_metrics["macro_f1"]
            balanced_accuracy = normal_metrics["balanced_accuracy"]
            
            if print_result:
                utils.print_color("\n--- Normal Examples (S/NS) Metrics ---", "yellow")
                utils.print_color(f"Macro F1: {macro_f1:.3f}", "green")
                utils.print_color(f"Balanced Accuracy: {balanced_accuracy:.3f}", "green")
        
        # Evaluate ambiguous examples
        ambiguous_metrics = {}
        if ambiguous_results:
            s_count = sum(1 for r in ambiguous_results if r["prediction"] == FactVerificationLabel.SUPPORTED)
            ns_count = sum(1 for r in ambiguous_results if r["prediction"] == FactVerificationLabel.NOT_SUPPORTED)
            parsing_errors_ambig = sum(1 for r in ambiguous_results if r["prediction"] == FactVerificationLabel.PARSING_ERROR)
            
            valid_ambiguous_count = len(ambiguous_results) - parsing_errors_ambig
            s_pct = s_count/valid_ambiguous_count*100 if valid_ambiguous_count > 0 else 0.0
            ns_pct = ns_count/valid_ambiguous_count*100 if valid_ambiguous_count > 0 else 0.0
            
            ambiguous_metrics = {
                "s_count": s_count,
                "s_percentage": s_pct,
                "ns_count": ns_count,
                "ns_percentage": ns_pct,
                "parsing_errors": parsing_errors_ambig,
                "valid_count": valid_ambiguous_count
            }
            
            if print_result:
                utils.print_color("\n--- Grayfacts Prediction Distribution ---", "yellow")
                utils.print_color(f"Valid ambiguous examples: {valid_ambiguous_count}", "green")
                utils.print_color(f"Predicted as Supported: {s_count} ({s_pct:.1f}%)", "green")
                utils.print_color(f"Predicted as Not Supported: {ns_count} ({ns_pct:.1f}%)", "green")
                if parsing_errors_ambig > 0:
                    utils.print_color(f"Parsing errors: {parsing_errors_ambig} ({parsing_errors_ambig/len(ambiguous_results)*100:.1f}%)", "green")
        
        # Key metrics for grayfacts - prioritize original labels if available
        if print_result:
            utils.print_color("\nKey Metrics for Grayfacts:", "yellow")
            utils.print_color(f"Macro F1 on Original Labels: {original_macro_f1:.3f}", "green")
            utils.print_color(f"Ambiguous Supported %: {s_pct:.1f}%", "green")
            utils.print_color(f"Total Parsing Errors: {total_parsing_errors}", "green")
        
        return {
            "dataset_type": "grayfacts",
            "num_results": num_results,
            "num_normal": num_normal,
            "num_ambiguous": num_ambiguous,
            "total_parsing_errors": total_parsing_errors,
            "normal_metrics": normal_metrics,
            "ambiguous_metrics": ambiguous_metrics,
            "original_metrics": original_metrics,
            "macro_f1": macro_f1,
            "original_macro_f1": original_macro_f1,
            "original_s_count": original_s_count,
            "original_ns_count": original_ns_count,
            "ambiguous_supported_pct": s_pct,
            "has_original_labels": len(results_with_original_labels) > 0,
        }

    def report(self, category_wise: bool = True, topic_wise: bool = True) -> None:
        """Generate comprehensive report based on dataset type"""
        dataset_type = self._get_dataset_type()
        
        if dataset_type == "clearfacts":
            # Category-wise report for clearfacts
            if category_wise:
                all_categories = sorted(
                    list(set([result["category"] for result in self.results]))
                )
                utils.print_color("\n=== CATEGORY-WISE RESULTS ===", "blue")
                for category in all_categories:
                    category_results = [
                        result for result in self.results if result["category"] == category
                    ]
                    utils.print_color(f"\nCategory: {category}", "yellow")
                    self._report_clearfacts(category_results)

            # Topic-wise report for clearfacts
            if topic_wise:
                all_topics = sorted(
                    list(set([result["topic"] for result in self.results]))
                )
                if len(all_topics) > 1:  # Only show if multiple topics
                    utils.print_color("\n=== TOPIC-WISE RESULTS ===", "blue")
                    for topic in all_topics:
                        topic_results = [
                            result for result in self.results if result["topic"] == topic
                        ]
                        utils.print_color(f"\nTopic: {topic}", "yellow")
                        self._report_clearfacts(topic_results)

            # Overall report
            utils.print_color("\n=== OVERALL RESULTS ===", "blue")
            self._report_clearfacts(self.results)
            
        else:  # grayfacts
            # Category-wise report for grayfacts
            if category_wise:
                all_categories = sorted(
                    list(set([result["category"] for result in self.results]))
                )
                utils.print_color("\n=== CATEGORY-WISE RESULTS ===", "blue")
                for category in all_categories:
                    category_results = [
                        result for result in self.results if result["category"] == category
                    ]
                    utils.print_color(f"\nCategory: {category}", "yellow")
                    self._report_grayfacts(category_results)

            # Topic-wise report for grayfacts
            if topic_wise:
                all_topics = sorted(
                    list(set([result["topic"] for result in self.results]))
                )
                if len(all_topics) > 1:  # Only show if multiple topics
                    utils.print_color("\n=== TOPIC-WISE RESULTS ===", "blue")
                    for topic in all_topics:
                        topic_results = [
                            result for result in self.results if result["topic"] == topic
                        ]
                        utils.print_color(f"\nTopic: {topic}", "yellow")
                        self._report_grayfacts(topic_results)

            # Overall report
            utils.print_color("\n=== OVERALL RESULTS ===", "blue")
            self._report_grayfacts(self.results)

    def report_to_dict(
        self, category_wise: bool = True, topic_wise: bool = True
    ) -> dict:
        """Generate report as dictionary for programmatic access"""
        dataset_type = self._get_dataset_type()
        
        if dataset_type == "clearfacts":
            results = {"all": self._report_clearfacts(self.results, print_result=False)}
        else:
            results = {"all": self._report_grayfacts(self.results, print_result=False)}
        
        if category_wise:
            all_categories = sorted(
                list(set([result["category"] for result in self.results]))
            )
            for category in all_categories:
                category_results = [
                    result for result in self.results if result["category"] == category
                ]
                if dataset_type == "clearfacts":
                    results[category] = self._report_clearfacts(category_results, print_result=False)
                else:
                    results[category] = self._report_grayfacts(category_results, print_result=False)
                    
        if topic_wise:
            all_topics = sorted(
                list(set([result["topic"] for result in self.results]))
            )
            for topic in all_topics:
                topic_results = [
                    result for result in self.results if result["topic"] == topic
                ]
                topic_key = f"topic_{topic}"
                if dataset_type == "clearfacts":
                    results[topic_key] = self._report_clearfacts(topic_results, print_result=False)
                else:
                    results[topic_key] = self._report_grayfacts(topic_results, print_result=False)

        return results

    def save(self, output_dir: str) -> None:
        def custom_encoder(obj: object) -> object:
            if isinstance(obj, FactVerificationLabel):
                return obj.value
            return obj

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for result in self.results:
                f.write(json.dumps(result, default=custom_encoder) + "\n")

    def load(
        self,
        output_dir: str,
        type: str = None,
    ) -> None:
        self.results = []

        def custom_decoder(obj: dict) -> dict:
            if "prediction" in obj:
                obj["prediction"] = FactVerificationLabel(obj["prediction"])
            if "label" in obj:
                obj["label"] = FactVerificationLabel(obj["label"])
            return obj

        with open(os.path.join(output_dir, "results.jsonl"), "r") as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                line = json.loads(line, object_hook=custom_decoder)
                if line["prediction"] == FactVerificationLabel.PARSING_ERROR and (
                    line["response"].startswith("yes")
                    or line["response"].startswith("no")
                ):
                    line["prediction"] = (
                        FactVerificationLabel.SUPPORTED
                        if line["response"].startswith("yes")
                        else FactVerificationLabel.NOT_SUPPORTED
                    )
                self.results.append(line)

    def evaluate_with_ambiguous(self, print_result: bool = True, report_granularity: str = "category") -> dict:
        """Legacy method for backward compatibility - now delegates to report_grayfacts"""
        return self._report_grayfacts(self.results, print_result)


class FactVerifierBase(abc.ABC):

    def __init__(self, num_processes: int = 1):
        self.num_processes = num_processes

    @staticmethod
    def get_bm25_passages(
        topic: str, query: str, passages: list[str], k: int
    ) -> list[str]:
        """
        Get top k passages using BM25 algorithm.
        """
        bm25 = rank_bm25.BM25Okapi(
            [psg.replace("<s>", "").replace("</s>", "").split() for psg in passages]
        )
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    @abc.abstractmethod
    def _fact_verification(
        self, data: FactVerificationDataFormat
    ) -> tuple[FactVerificationLabel, str]:
        raise NotImplementedError

    def run(self, dataset: list[FactVerificationDataFormat]) -> FactVerificationResult:
        fact_verification_result = FactVerificationResult()
        if self.num_processes > 1:
            with mp.Pool(self.num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._fact_verification, dataset),
                        total=len(dataset),
                        desc="Running Fact Verification...",
                    )
                )
        else:
            results = []
            for data in tqdm(dataset, desc="Running Fact Verification..."):
                results.append(self._fact_verification(data))

        for data, (result, response) in zip(dataset, results):
            fact_verification_result.add_result(
                data, prediction=result, response=response
            )

        return fact_verification_result


class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        show_responses: bool = False,
        show_prompts: bool = False,
    ) -> None:
        """Initializes a model."""
        is_avior = False
        if "OPENAI:" in model_name:
            model_name = model_name.replace("OPENAI:", "")
        elif "AVIOR:" in model_name:
            model_name = model_name.replace("AVIOR:", "")
            is_avior = True

        self.is_avior = is_avior
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_responses = show_responses
        self.show_prompts = show_prompts

    def generate(
        self, prompt: str, do_debug: bool = False, return_log_probs: bool = False
    ) -> str | tuple[str, list[tuple[str, float]]]:
        """Generates a response to a prompt."""
        message = [{"role": "user", "content": prompt}]
        try:
            if self.is_avior:
                output = get_avior_chat_response_sync(
                    prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=return_log_probs,
                )
            else:
                output = get_openai_chat_response_sync(
                    prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=return_log_probs,
                )
        except Exception as e:
            print(f"Error generating response: {e}. Query: {message}")
            return ""
        response = output.choices[0].message.content.strip()
        if do_debug:
            if self.show_prompts:
                utils.print_color(prompt, "magenta")
            if self.show_responses:
                utils.print_color(response, "cyan")
        if return_log_probs:
            log_probs = output.choices[0].logprobs.content  # list of logprobs.
            log_probs = [(lp.token, lp.logprob) for lp in log_probs]
            return response, log_probs

        return response

class GeminiModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        show_responses: bool = False,
        show_prompts: bool = False,
    ) -> None:
        """Initializes a Gemini model."""
        if "GEMINI:" in model_name:
            model_name = model_name.replace("GEMINI:", "")
            
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_responses = show_responses
        self.show_prompts = show_prompts

    def generate(
        self, prompt: str, do_debug: bool = False, return_log_probs: bool = False
    ) -> str | tuple[str, list[tuple[str, float]]]:
        """Generates a response to a prompt."""
        try:
            # Convert to message format with user role, matching OpenAI pattern
            messages = [{"role": "user", "content": prompt}]
            output = get_gemini_chat_response_sync(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            print(f"Error generating response: {e}. Query: {prompt}")
            return ""
            
        response = output.strip()
        
        if do_debug:
            if self.show_prompts:
                utils.print_color(prompt, "magenta")
            if self.show_responses:
                utils.print_color(response, "cyan")
                
        if return_log_probs:
            # Gemini API doesn't support log probs currently
            # Return empty list for compatibility
            return response, []
            
        return response


class AnthropicModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        show_responses: bool = False,
        show_prompts: bool = False,
    ) -> None:
        """Initializes an Anthropic model."""
        if "ANTHROPIC:" in model_name:
            model_name = model_name.replace("ANTHROPIC:", "")
            
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_responses = show_responses
        self.show_prompts = show_prompts

    def generate(
        self, prompt: str, do_debug: bool = False, return_log_probs: bool = False
    ) -> str | tuple[str, list[tuple[str, float]]]:
        """Generates a response to a prompt."""
        try:
            # Convert to message format with user role, matching OpenAI pattern
            messages = [{"role": "user", "content": prompt}]
            output = get_anthropic_chat_response_sync(
                messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            print(f"Error generating response: {e}. Query: {prompt}")
            return ""
            
        response = output.strip()
        
        if do_debug:
            if self.show_prompts:
                utils.print_color(prompt, "magenta")
            if self.show_responses:
                utils.print_color(response, "cyan")
                
        if return_log_probs:
            # Anthropic API doesn't support log probs currently
            # Return empty list for compatibility
            return response, []
            
        return response