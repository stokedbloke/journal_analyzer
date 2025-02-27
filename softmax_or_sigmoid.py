import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model
model_name = "SamLowe/roberta-base-go_emotions"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check model's problem type (if defined)
if hasattr(model.config, "problem_type"):
    print("Model problem type:", model.config.problem_type)
    if model.config.problem_type in ["single_label_classification", "multi_class"]:
        print("The model uses softmax for multi-class classification.")
    elif model.config.problem_type in ["multi_label_classification"]:
        print("The model uses sigmoid for multi-label classification.")

# Sample text
text = "I felt both happy and sad today."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Get raw logits from the model
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Check if softmax or sigmoid is applied
probabilities_softmax = torch.nn.functional.softmax(logits, dim=-1)
probabilities_sigmoid = torch.nn.functional.sigmoid(logits)

# Print results
print("Sum of softmax probabilities:", probabilities_softmax.sum().item())
print("Sum of sigmoid probabilities:", probabilities_sigmoid.sum().item())
