from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    
    # Tokenize the input
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    
    # Generate a response
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
