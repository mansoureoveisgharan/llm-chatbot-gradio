import gradio as gr
import ollama
from typing import Generator

# Global chat history to maintain conversation context
chat_history = []

def chatbot(prompt: str, model: str) -> Generator[str, None, None]:
    """
    Handle chat interaction with Ollama models.
    
    Args:
        prompt: User input text
        model: Selected model name (e.g., gemma3:1b)
    
    Yields:
        Streaming response from the LLM
    """
    global chat_history
    
    # Validate input
    if not prompt or not prompt.strip():
        yield "❌ Please enter a valid question."
        return
    
    # Add user message to history
    chat_history.append({"role": "user", "content": prompt})
    
    # Build messages with system prompt (only once at start)
    if len(chat_history) == 1:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ] + chat_history
    else:
        messages = chat_history
    
    try:
        # Stream response from Ollama
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        
        # Collect and yield streaming response
        response = ""
        for chunk in stream:
            if chunk and chunk.message and chunk.message.content:
                response += chunk.message.content
                yield response
        
        # Save assistant response to history after completion
        if response:
            chat_history.append({"role": "assistant", "content": response})
        
        # Limit history to prevent memory issues (keep last 20 exchanges = 40 messages)
        if len(chat_history) > 40:
            # Keep system message if exists, plus last 39 messages
            if chat_history[0]["role"] == "system":
                chat_history = [chat_history[0]] + chat_history[-39:]
            else:
                chat_history = chat_history[-40:]
                
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n💡 Make sure model '{model}' is installed: `ollama pull {model}`"
        yield error_msg


# UI Components
prompt_text = gr.Textbox(
    label="Your Question",
    placeholder="Type your message here...",
    lines=7
)

response_llm = gr.Markdown(
    label="AI Response"
)

model_selector = gr.Dropdown(
    choices=["gemma3:1b", "gemma2:2b"],
    label="Select Model",
    value="gemma3:1b"
)

# Build interface
gr.Interface(
    fn=chatbot,
    inputs=[prompt_text, model_selector],
    outputs=response_llm,
    flagging_mode="never"
).launch(inbrowser=True)