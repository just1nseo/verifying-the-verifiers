"""
This file is copied and modified from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a.
Thanks to Graham Neubig for sharing the original code.
"""

import os
from common.shared_config import anthropic_api_key

# Set environment variable ANTHROPIC_API_KEY
if os.getenv("ANTHROPIC_API_KEY") is None:
    print("Setting ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
else:
    print("ANTHROPIC_API_KEY already set")

# Import Anthropic client with graceful error handling
try:
    from anthropic import AsyncAnthropic, Anthropic
    # Initialize Anthropic clients
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    aclient = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    print("Anthropic SDK not installed. To use Anthropic models, install with: pip install anthropic")
    AsyncAnthropic = None
    Anthropic = None
    client = None
    aclient = None

import asyncio
from typing import Any, Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=10, min=30, max=120),
    reraise=True,
)
def get_anthropic_response_sync(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    **completion_kwargs
) -> str:
    """
    Get a response from the Anthropic API synchronously with retry logic.
    
    Args:
        prompt: The prompt to send to the model
        model: The model name to use (e.g., claude-3-5-sonnet-20241022)
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **completion_kwargs: Additional arguments for the API call
        
    Returns:
        The generated text response
    """
    if client is None:
        raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")
    
    try:
        # Anthropic uses a messages-based API
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **completion_kwargs
        )
        
        return message.content[0].text
        
    except Exception as e:
        # Fallback without additional kwargs if there's an error
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return message.content[0].text


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=10, min=30, max=120),
    reraise=True,
)
def get_anthropic_chat_response_sync(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    **completion_kwargs
) -> str:
    """
    Get a response from the Anthropic API using the chat format with retry logic.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model name to use
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **completion_kwargs: Additional arguments for the API call
        
    Returns:
        The generated text response
    """
    if client is None:
        raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")
    
    try:
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Anthropic handles system messages separately
                system_message = content
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                anthropic_messages.append({
                    "role": "assistant", 
                    "content": content
                })
        
        # Create the API call
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
            **completion_kwargs
        }
        
        # Add system message if present
        if system_message:
            kwargs["system"] = system_message
        
        response = client.messages.create(**kwargs)
        
        return response.content[0].text
        
    except Exception as e:
        # Fallback without additional kwargs if there's an error
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = client.messages.create(**kwargs)
        return response.content[0].text


async def dispatch_anthropic_chat_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    **completion_kwargs: Any,
) -> List[Any]:
    """Dispatches requests to Anthropic messages API asynchronously.

    Args:
        messages_list: List of messages to be sent to Anthropic messages API.
        model: Anthropic model to use.
        completion_kwargs: Keyword arguments to be passed to Anthropic messages API.
    Returns:
        List of responses from Anthropic API.
    """
    if aclient is None:
        raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")
    
    async_responses = [
        aclient.messages.create(model=model, messages=x, **completion_kwargs)
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_anthropic_prompt_requests(
    prompt_list: List[str],
    model: str,
    **completion_kwargs: Any,
) -> List[Any]:
    """Dispatches requests to Anthropic messages API asynchronously using simple prompts.

    Args:
        prompt_list: List of prompts to be sent to Anthropic messages API.
        model: Anthropic model to use.
        completion_kwargs: Keyword arguments to be passed to Anthropic messages API.
    Returns:
        List of responses from Anthropic API.
    """
    if aclient is None:
        raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")
    
    # Convert prompts to message format
    messages_list = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    
    async_responses = [
        aclient.messages.create(model=model, messages=messages, **completion_kwargs)
        for messages in messages_list
    ]
    return await asyncio.gather(*async_responses)


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=10, min=30, max=120),
    reraise=True,
)
def dispatch_batch_anthropic_requests_with_retry(
    message_or_prompt_batch: list,
    model: str,
    dispatch_func: Callable,
    **completion_kwargs,
) -> list[Any]:
    return asyncio.run(
        dispatch_func(message_or_prompt_batch, model=model, **completion_kwargs)
    ) 