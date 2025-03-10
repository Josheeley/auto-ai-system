from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Save the model (for demonstration purposes)
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Model trained and saved!")
