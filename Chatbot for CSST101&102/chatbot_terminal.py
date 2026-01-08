import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the model and data files
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json', 'rb'))

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, show_details=True):
    """Create bag of words array from sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the sentence"""
    p = bow(sentence, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    """Get response based on predicted intent"""
    if len(ints) > 0:
        tag = ints[0]['intent']
        
        # Look for the intent in the intents dictionary
        # The intents.json structure has intent tags as keys directly
        if tag in intents_json:
            result = random.choice(intents_json[tag]['responses'])
        else:
            result = "I don't understand. Please ask me something related to fire safety."
        return result
    else:
        return "I don't understand. Please ask me something related to fire safety."

def chatbot_response(msg):
    """Main function to get chatbot response"""
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res

def main():
    print("Fire Safety Chatbot is ready! Type 'quit' to exit.")
    print("Hi there! How can I assist you with fire safety questions?")
    
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            print("Goodbye! Remember to always practice fire safety.")
            break
        response = chatbot_response(message)
        print(f"{response}")

if __name__ == "__main__":
    main()
