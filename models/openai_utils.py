"""
This file is copied and modified from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a.
Thanks to Graham Neubig for sharing the original code.
"""

import os

from common.shared_config import openai_api_key

# set environment variable OPENAI_API_KEY
if os.getenv("OPENAI_API_KEY") is None:
    print("Setting OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    print("OPENAI_API_KEY already set")

from openai import AsyncOpenAI, OpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion

aclient = AsyncOpenAI()
client = OpenAI()
avior_client = OpenAI(
    base_url="http://avior.mlfoundry.com/live-inference/v1",
    api_key=os.getenv("FOUNDRY_API_KEY", None),
)
import asyncio
from typing import Any, Callable, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=10, min=30, max=120),
    reraise=True,
)
def dispatch_batch_openai_requests_with_retry(
    message_or_prompt_batch: list,
    model: str,
    dispatch_func: Callable,
    **completion_kwargs,
) -> list[Any]:
    return asyncio.run(
        dispatch_func(message_or_prompt_batch, model=model, **completion_kwargs)
    )


async def dispatch_openai_chat_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    **completion_kwargs: Any,
) -> List[ChatCompletion]:
    """Dispatches requests to OpenAI chat completion API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI chat completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI ChatCompletion API. See https://platform.openai.com/docs/api-reference/chat for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        aclient.chat.completions.create(model=model, messages=x, **completion_kwargs)
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requests(
    prompt_list: List[str],
    model: str,
    **completion_kwargs: Any,
) -> List[Completion]:
    """Dispatches requests to OpenAI text completion API asynchronously.

    Args:
        prompt_list: List of prompts to be sent to OpenAI text completion API.
        model: OpenAI model to use.
        completion_kwargs: Keyword arguments to be passed to OpenAI text completion API. See https://platform.openai.com/docs/api-reference/completions for details.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        aclient.completions.create(model=model, prompt=x, **completion_kwargs)
        for x in prompt_list
    ]
    return await asyncio.gather(*async_responses)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=10, min=30, max=120),
    reraise=True,
)
def get_openai_chat_response_sync(
    prompt: str, model: str, **completion_kwargs
) -> ChatCompletion:
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            **completion_kwargs,
        )
    except Exception as e:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )

    return chat_completion


def get_avior_chat_response_sync(
    prompt: str, model: str, **completion_kwargs
) -> ChatCompletion:
    chat_completion = avior_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, **completion_kwargs
    )
    return chat_completion


if __name__ == "__main__":
    chat_completion_responses = asyncio.run(
        dispatch_openai_chat_requests(
            messages_list=[
                [
                    {
                        "role": "user",
                        "content": "Write a poem about asynchronous execution.",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Write a poem about asynchronous pirates.",
                    }
                ],
            ],
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,
        )
    )

    for i, x in enumerate(chat_completion_responses):
        print(f"Chat completion response {i}:\n{x.choices[0].message.content}\n\n")

    prompt_completion_responses = asyncio.run(
        dispatch_openai_prompt_requests(
            prompt_list=[
                "Write a poem about asynchronous execution.\n",
                "Write a poem about asynchronous pirates.\n",
            ],
            model="text-davinci-003",
            temperature=0.3,
            max_tokens=200,
            top_p=1.0,
        )
    )

    for i, x in enumerate(prompt_completion_responses):
        print(f"Prompt completion response {i}:\n{x.choices[0].text}\n\n")
