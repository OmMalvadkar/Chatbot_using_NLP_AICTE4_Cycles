import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Apply gradient background styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set up NLTK data path and download required data
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r"C:\Users\ommal\OneDrive\Desktop\AICTE_INTERSHIP_PROJECT\intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("🤖 Intents of Chatbot using NLP")
    menu = ["🏠 Home", "📜 Conversation History", "ℹ️ About"]
    choice = st.sidebar.selectbox("📌 Menu", menu)

    if choice == "🏠 Home":
        st.write("Type a message and press **Enter** to chat!")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("✏️ You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Display chat messages in a structured format
            with st.chat_message("user"):
                st.markdown(f"**You:** {user_input}")
            
            with st.chat_message("assistant"):
                st.markdown(f"**Chatbot:** {response}")

            # Save chat history
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("👋 Thanks for chatting! Have a great day!")
                st.stop()

    elif choice == "📜 Conversation History":
        st.header("📜 Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    with st.chat_message("user"):
                        st.markdown(f"**User:** {row[0]}")
                    with st.chat_message("assistant"):
                        st.markdown(f"**Chatbot:** {row[1]}")
        except FileNotFoundError:
            st.error("No conversation history found.")

     # About Menu
    elif choice == "ℹ️ About":
        st.markdown("<h3 style='color:#ffffff;'>ℹ️ About This Project</h3>", unsafe_allow_html=True)
        st.write("""
        The goal of this project is to create a chatbot that understands and responds to user input based on **intents**. It is built using:
        - **Natural Language Processing (NLP)**
        - **Logistic Regression**
        - **Streamlit for Web UI**
        """)

        st.subheader("📌 Project Overview:")
        st.write("""
        🔹 The chatbot is trained on labeled intents and uses **TF-IDF Vectorization** + **Logistic Regression** to classify user input.
        🔹 The chatbot interface is built using **Streamlit**, allowing users to interact with it easily.
        """)

        st.subheader("📚 Dataset:")
        st.write("""
        The dataset contains:
        - **Intents** (e.g., "greeting", "budget", "about")
        - **Entities** (e.g., "Hi", "How do I create a budget?", "What is your purpose?")
        - **Patterns** (User input examples mapped to intents)
        """)

        st.subheader("🎨 Chatbot UI Features:")
        st.write("""
        ✅ **Interactive Web Chat Interface**  
        ✅ **User-Friendly Design**  
        ✅ **Chat History Logging**  
        ✅ **Response Generation using NLP**
        """)

        st.subheader("🔍 Future Improvements:")
        st.write("""
        🔹 Improve chatbot accuracy with **Deep Learning (e.g., LSTMs, Transformers)**  
        🔹 Implement **Context Awareness** for more dynamic responses  
        🔹 Extend the dataset with more intents & responses  
        """)
if __name__ == '__main__':
    main()