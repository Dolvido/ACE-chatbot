import os
from typing import Optional, Tuple
from threading import Lock
import gradio as gr
from query_data import get_basic_qa_chain, get_custom_prompt_qa_chain, get_condense_prompt_qa_chain

def select_qa_chain(prompt):
    if len(prompt.split()) < 10:  # Example condition based on the prompt length
        return get_basic_qa_chain()
    elif "specific keyword" in prompt.lower():  # Example condition based on keywords in the prompt
        return get_custom_prompt_qa_chain()
    else:
        return get_condense_prompt_qa_chain()


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
        self.chain = get_basic_qa_chain()
        self.log_folder = "logs/message-history"
        os.makedirs(self.log_folder, exist_ok=True)  # Create the folder if it doesn't exist

    def log_conversation(self, question, answer):
        with open(os.path.join(self.log_folder, "conversation_log.txt"), "a") as file:
            file.write(f"Q: {question}\nA: {answer}\n\n")

    def __call__(self, inp: str, history: Optional[Tuple[str, str]]):
        self.lock.acquire()
        try:
            history = history or []
                
            if self.chain is None:
                answer = "Chain is not available"
                history.append((inp, answer))
                self.log_conversation(inp, answer)
                return history, history
                
            answer = self.chain({"question": inp})["answer"]
            history.append((inp, answer))
            self.log_conversation(inp, answer)
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

# Note: The hex color codes in the CSS below are invalid (##), fixing them
custom_css = """
    .gradio-container {
        background-color: #117dac;
    }
    .gr-textbox {
        background-color: #b33505;
        color: #000000;
    }
    .gr-button {
        background-color: #4CAF50;
        color: white;
    }
"""

block = gr.Blocks(css=custom_css)

with block:
    with gr.Row():
        gr.Markdown("<h3><center>ACE chat (ACE Internal State)</center></h3>")

    chatbot = gr.Chatbot(height=800)

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about the internal state of the ACE",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(
            full_width=False)

    gr.Examples(
        examples=[
            "What is the meaning of life?",
            "What is the purpose of life?",
            "What is the purpose of the ACE?"
        ],
        inputs=message,
    )

    gr.HTML("Demo application of an Autonomous Cognitive Entity.")

    state = gr.State()
    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=True)
