import os
from common.shared_config import gemini_api_key

# Set environment variable GEMINI_API_KEY
if os.getenv("GEMINI_API_KEY") is None:
    print("Setting GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = gemini_api_key
else:
    print("GEMINI_API_KEY already set")

import google.generativeai as genai
from typing import Optional, List, Dict, Any

# Initialize the Gemini API with your API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_gemini_response_sync(
    prompt: str,
    model: str = "gemini-1.5-pro",
    temperature: float = 0.5,
    max_tokens: int = 2048,
) -> str:
    """
    Get a response from the Gemini API synchronously.
    
    Args:
        prompt: The prompt to send to the model
        model: The model name to use
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The generated text response
    """
    try:
        # Configure the model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 1.0,
            "top_k": 40,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        # Check if the prompt looks like it might be in a chat format
        # This is a simple heuristic - if it contains "user:" or similar markers
        if "user:" in prompt.lower() or "assistant:" in prompt.lower() or "system:" in prompt.lower():
            # Try to parse it into a chat format
            chat = genai.ChatSession(model=model)
            
            # Simple parsing of the prompt into chat turns
            lines = prompt.split("\n")
            current_role = None
            current_content = []
            
            for line in lines:
                if line.lower().startswith("user:"):
                    if current_role and current_content:
                        chat.send_message("\n".join(current_content))
                    current_role = "user"
                    current_content = [line[5:].strip()]
                elif line.lower().startswith("assistant:"):
                    if current_role and current_content:
                        chat.send_message("\n".join(current_content))
                    current_role = "assistant"
                    current_content = [line[10:].strip()]
                elif line.lower().startswith("system:"):
                    # System messages are handled differently in Gemini
                    # We'll prepend them to the next user message
                    if current_role and current_content:
                        chat.send_message("\n".join(current_content))
                    current_role = "system"
                    current_content = [line[7:].strip()]
                else:
                    if current_role:
                        current_content.append(line)
            
            # Send the final message if there is one
            if current_role and current_content:
                response = chat.send_message("\n".join(current_content))
                return response.text
        
        # If not in chat format or chat parsing failed, use standard generation
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        raise e

def get_gemini_chat_response_sync(
    messages: List[Dict[str, str]],
    model: str = "gemini-1.5-pro",
    temperature: float = 0.5,
    max_tokens: int = 2048,
) -> str:
    """
    Get a response from the Gemini API using the chat format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model name to use
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The generated text response
    """
    try:
        # Configure the model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 1.0,
            "top_k": 40,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        # Create a chat session
        chat = genai.ChatSession(model=model)
        
        # Process each message
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                if message == messages[-1]:  # If this is the last message
                    response = chat.send_message(content)
                    return response.text
                else:
                    chat.send_message(content)
            elif role == "assistant":
                # For assistant messages in the history, we need to add them to the chat
                # This simulates the assistant's previous responses
                chat.history.append({"role": "model", "parts": [content]})
            elif role == "system":
                # Gemini doesn't have a direct system message concept
                # We'll prepend it to the next user message
                next_user_idx = next((i for i, m in enumerate(messages[messages.index(message)+1:], 
                                     messages.index(message)+1) if m["role"] == "user"), None)
                if next_user_idx is not None:
                    messages[next_user_idx]["content"] = f"System: {content}\n\nUser: {messages[next_user_idx]['content']}"
        
        # If we get here without returning, it means there was no user message at the end
        # In this case, we'll just send a final empty message to get a response
        response = chat.send_message("")
        return response.text
        
    except Exception as e:
        print(f"Error in Gemini chat API call: {e}")
        raise e 