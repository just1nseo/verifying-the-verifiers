# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared configuration across all project code."""

import os

################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# prompt_postamble: str = The postamble to seek more details in output.
# openai_api_key: str = OpenAI API key.
# anthropic_api_key: str = Anthropic API key.
# gemini_api_key: str = Gemini API key.
# random_seed: int = random seed to use across codebase.
# model_options: Dict[str, str] = mapping from short model name to full name.
# model_string: Dict[str, str] = mapping from short model name to saveable name.
# task_options: Dict[str, Any] = mapping from short task name to task details.
# root_dir: str = path to folder containing all files for this project.
# path_to_data: str = directory storing task information.
# path_to_result: str = directory to output results.
################################################################################
openai_api_key = ''
anthropic_api_key = ''
gemini_api_key = ''
random_seed = 1
model_options = {
    'o3': 'OPENAI:o3-2025-04-16',
    'o1': 'OPENAI:o1-2024-12-17',
    'o1_preview': 'OPENAI:o1-preview-2024-09-12',
    'o3_mini': 'OPENAI:o3-mini-2025-01-31',
    'o1_mini': 'OPENAI:o1-mini-2024-09-12',
    'gpt_4_1': 'OPENAI:gpt-4.1-2025-04-14',
    'gpt_4_1_mini': 'OPENAI:gpt-4.1-mini-2025-04-14',
    'gpt_4_1_nano': 'OPENAI:gpt-4.1-nano-2025-04-14',
    'gpt_4o_mini': 'OPENAI:gpt-4o-mini-2024-07-18',
    'gpt_4o': 'OPENAI:gpt-4o-2024-08-06',
    'gpt_4_turbo': 'OPENAI:gpt-4-0125-preview',
    'gpt_4': 'OPENAI:gpt-4-0613',
    'gpt_4_32k': 'OPENAI:gpt-4-32k-0613',
    'gpt_35_turbo': 'OPENAI:gpt-3.5-turbo-0125',
    'gpt_35_turbo_16k': 'OPENAI:gpt-3.5-turbo-16k-0613',
    'claude_4_opus': 'ANTHROPIC:claude-opus-4-20250514',
    'claude_4_sonnet': 'ANTHROPIC:claude-sonnet-4-20250514',
    'claude_3_7_sonnet': 'ANTHROPIC:claude-3-7-sonnet-20250219',
    'claude_3_5_haiku': 'ANTHROPIC:claude-3-5-haiku-20241022',
    'gemini_2_5_pro': 'GEMINI:gemini-2.5-pro-preview-05-06',
    'gemini_2_5_flash': 'GEMINI:gemini-2.5-flash-preview-05-20',
    'gemini_2_0_flash': 'GEMINI:gemini-2.0-flash',
    'gemini_2_0_flash_lite': 'GEMINI:gemini-2.0-flash-lite',
}
model_string = {
    'o1_preview': 'o1-preview-2024-09-12',
    'o1_mini': 'o1-mini-2024-09-12',
    'gpt_4_mini': 'gpt4mini',
    'gpt_4_turbo': 'gpt4turbo',
    'gpt_4': 'gpt4',
    'gpt_4o': 'gpt4o',
    'gpt_4_32k': 'gpt432k',
    'gpt_35_turbo': 'gpt35turbo',
    'gpt_35_turbo_16k': 'gpt35turbo16k',
    'claude_3_opus': 'claude3opus',
    'claude_3_sonnet': 'claude3sonnet',
    'claude_21': 'claude21',
    'claude_20': 'claude20',
    'claude_instant': 'claudeinstant',
    'gemini_1_5_pro': 'gemini15pro',
    'gemini_1_5_flash': 'gemini15flash',
    'gemini_1_0_pro': 'gemini10pro',
}
task_options = {}
root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2])
path_to_data = 'datasets/'
path_to_result = 'results/'
gemini_api_key = ''
