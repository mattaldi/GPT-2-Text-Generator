import gradio as gr
from transformers import pipeline, set_seed

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2')
set_seed(42)  # for reproducibility.

def generate_text(prompt, max_length, temperature):
    # Generate text based on the input prompt
    generated_text = generator(prompt, max_length=max_length, temperature=temperature, num_return_sequences=1)[0]['generated_text']
    return generated_text

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Maximum Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT-2 Text Generator",
    description="Enter a prompt and generate text using GPT-2.",
)

# Launch the app
iface.launch()
