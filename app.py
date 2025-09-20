import gradio as gr
from unsloth import FastLanguageModel
import torch
import time

# Global variables for model caching
model = None
tokenizer = None

def load_model():
    """Load the trained model once"""
    global model, tokenizer
    if model is None:
        print("Loading Gemma Career Guidance model...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="./gemma_career_final",
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True

def generate_career_response(message, history):
    """Generate career guidance response"""
    if not load_model():
        return "Sorry, I'm having trouble loading. Please try again in a moment."

    # Handle empty messages
    if not message.strip():
        return "Please ask me a career-related question!"

    try:
        # Format the conversation prompt
        prompt = f"""<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
"""

        # Generate response
        inputs = tokenizer([prompt], return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and clean the response
        response = tokenizer.batch_decode(outputs)[0]
        response = response.split('<start_of_turn>model')[-1]

        if '<end_of_turn>' in response:
            response = response.split('<end_of_turn>')[0]

        response = response.strip()

        # Fallback for empty responses
        if not response:
            response = "I'd be happy to help with your career question. Could you please provide more details?"

        return response

    except Exception as e:
        print(f"Generation error: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try rephrasing your question."

# Create custom CSS for professional look
css = """
#chatbot {
    height: 600px !important;
}

.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}

#component-0 {
    height: 100vh !important;
}

.message {
    font-size: 16px !important;
}

.bot-message {
    background-color: #f0f8ff !important;
    border-left: 4px solid #007bff !important;
    padding-left: 10px !important;
}

.user-message {
    background-color: #f8f9fa !important;
    border-left: 4px solid #28a745 !important;
    padding-left: 10px !important;
}
"""

# Sample conversation examples
examples = [
    ["What skills do I need to become a data scientist?"],
    ["How should I prepare for a software engineering interview?"],
    ["What's the best career path for someone interested in AI?"],
    ["How do I transition from marketing to product management?"],
    ["What certifications are valuable for cybersecurity careers?"],
    ["How do I negotiate salary in my first job?"]
]

# Create the Gradio interface
with gr.Blocks(css=css, title="Career Guidance AI Assistant") as demo:
    gr.Markdown(
        """
        # 🚀 Career Guidance AI Assistant

        Welcome to your personal career advisor! I'm here to help you with:
        - **Career planning** and guidance
        - **Interview preparation** tips and strategies  
        - **Skill development** recommendations
        - **Job search** strategies and advice
        - **Career transitions** and pivots
        - **Professional development** planning

        Ask me anything about your career journey! 💼
        """
    )

    # Main chatbot interface
    chatbot = gr.ChatInterface(
        generate_career_response,
        chatbot=gr.Chatbot(
            elem_id="chatbot",
            height=600,
            show_label=True,
            show_copy_button=True,
            bubble_full_width=False,
            avatar_images=("👨‍💼", "🤖")
        ),
        textbox=gr.Textbox(
            placeholder="Ask me about career planning, job interviews, skills, or anything career-related...",
            container=False,
            scale=7
        ),
        title=None,  # Already added above
        theme="soft",
        cache_examples=True,
        retry_btn="🔄 Retry",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear Chat"
    )

    # Add example questions
    gr.Markdown("### 💡 Try these example questions:")
    gr.Examples(
        examples=examples,
        inputs=chatbot.textbox,
        label="Click any example to get started:"
    )

    # Footer information
    gr.Markdown(
        """
        ---
        **🔒 Privacy**: Your conversations are not stored. Each session is independent.

        **⚡ Powered by**: Gemma-2-2b fine-tuned for career guidance | Built with ❤️ using Unsloth & Gradio

        **📝 Note**: This AI provides general career guidance. For specific situations, consider consulting with career professionals.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False  # Will be handled by HF Spaces
    )
