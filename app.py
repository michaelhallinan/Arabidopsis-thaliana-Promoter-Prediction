import gradio as gr
import numpy as np
from preprocess import load_fasta, one_hot_encode, add_gc_content_to_data
from model import PromoterClassifier

# Load pre-trained model
classifier = PromoterClassifier(model_path="final_promoter_model.keras")

def classify_promoter(fasta_file=None, sequence_input=None):
    """Runs the model on an uploaded FASTA file or a directly input sequence."""
    if fasta_file:
        sequences, _ = load_fasta(fasta_file.name)
    elif sequence_input:
        sequences = [sequence_input]
    else:
        return "Please provide either a FASTA file or a sequence."

    one_hot = one_hot_encode(sequences)
    X_with_gc = add_gc_content_to_data(sequences, one_hot)

    predicted_class, prediction_scores = classifier.predict(X_with_gc)

    results = []
    for i in range(len(predicted_class)):
        likelihood = prediction_scores[i][0] * 100  # Convert score to percentage
        if predicted_class[i][0] == 1:  # Assuming 1 is promoter
            results.append(f"\U0001F44D Yes, Sequence {i+1} is likely a promoter! \U0001F49A Score: {likelihood:.2f}%")
        else:
            results.append(f"\U0001F44E No, Sequence {i+1} is not likely a promoter. \U0001F494 Score: {likelihood:.2f}%")

    return "\n".join(results)

# Define Gradio interface
demo = gr.Interface(
    fn=classify_promoter, 
    inputs=[
        gr.File(label="\U0001F4C2 Upload a FASTA file (optional)"),
        gr.Textbox(label="\U0001F4DD Or, enter a DNA sequence (optional)", lines=2)
    ], 
    outputs=gr.Textbox(label="\U0001F4C8 Prediction Results", elem_id="output-text"),
    title="\U0001F33F Arabidopsis Promoter Classifier",
    description="\U0001F3A8 Upload a FASTA file or enter a sequence to classify a sequence or sequences as likely or unlikely to be a promoter.",
    theme="default",  # Using default theme for more customization
    css="""
    body {
        background: linear-gradient(135deg, #fce4ec, #f8bbd0, #f48fb1);
    }
    .gradio-container {
        font-family: 'Poppins', sans-serif;
        text-align: center;
    }
    .gradio-title {
        font-size: 26px;
        font-weight: bold;
        color: #880e4f;
    }
    .gradio-description {
        font-size: 18px;
        color: #6a1b9a;
    }
    .gradio-button {
        background: linear-gradient(90deg, #f06292, #ec407a);
        color: white;
        border-radius: 10px;
        padding: 10px 15px;
    }
    .gradio-button:hover {
        background: linear-gradient(90deg, #ec407a, #d81b60);
    }
    .gradio-textbox {
        border: 2px solid #f06292;
        border-radius: 8px;
        padding: 10px;
    }
    #output-text {
        font-size: 18px;
        font-weight: bold;
        color: #880e4f;
        background: #fce4ec;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #f06292;
    }
    """
)

if __name__ == "__main__":
    demo.launch()
