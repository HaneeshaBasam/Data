from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cohere
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set your Cohere API key
cohere_client = cohere.Client("szQ2OWamGao2coIRYMKrMAXBGc7ySVbkfbnjTatB")

# Load the dataset
file_path = 'C:/Users/hanee/OneDrive/Desktop/datathon/Disease_symptom_and_patient_profile_dataset.csv'
dataset = pd.read_csv(file_path)

# Convert categorical variables to numeric values
label_enc = LabelEncoder()
for column in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']:
    dataset[column] = label_enc.fit_transform(dataset[column])

# Prepare feature variables (X) and target variable (y)
X = dataset[['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']]
y = dataset['Disease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Set a probability threshold for predicting a disease
PROBABILITY_THRESHOLD = 0.1

# Function to predict the disease with stricter checks
def predict_disease(inputs):
    # Define "normal" conditions
    if (inputs['fever'] == 'No' and
        inputs['cough'] == 'No' and
        inputs['fatigue'] == 'No' and
        inputs['breathing'] == 'No' and
        inputs['blood_pressure'] == 'Normal' and
        inputs['cholesterol'] == 'Normal'):
        # Return "All okay" if all inputs indicate normal conditions
        return None

    # If not normal, prepare the input data for the model
    input_data = pd.DataFrame({
        'Fever': [1 if inputs['fever'] == 'Yes' else 0],
        'Cough': [1 if inputs['cough'] == 'Yes' else 0],
        'Fatigue': [1 if inputs['fatigue'] == 'Yes' else 0],
        'Difficulty Breathing': [1 if inputs['breathing'] == 'Yes' else 0],
        'Age': [inputs['age']],
        'Gender': [1 if inputs['gender'] == 'Male' else 0],
        'Blood Pressure': [0 if inputs['blood_pressure'] == 'Low' else (1 if inputs['blood_pressure'] == 'Normal' else 2)],
        'Cholesterol Level': [0 if inputs['cholesterol'] == 'Low' else (1 if inputs['cholesterol'] == 'Normal' else 2)]
    })

    # Predict the probability of each disease
    probabilities = model.predict_proba(input_data)[0]
    
    # Find the highest probability and the corresponding predicted disease
    max_prob = max(probabilities)
    predicted_disease_index = probabilities.argmax()
    predicted_disease = model.classes_[predicted_disease_index]

    # If the highest probability is below the threshold or doesn't match, return "You are all okay"
    if max_prob < PROBABILITY_THRESHOLD:
        return None

    # Otherwise, return the predicted disease
    return predicted_disease



# Cohere integration for detailed info
def get_cohere_disease_info(disease_name):
    response = cohere_client.generate(
        model='command-xlarge',
        prompt=f"Give two points on the disease {disease_name} for each of the following:\n"
               "1. Symptoms\n"
               "2. Causes\n"
               "3. Types\n"
               "4. Self-care tips\n"
               "5. Diagnosis methods\n"
               "6. Helpline hospital numbers in major cities",
        max_tokens=500,
        temperature=0.7
    )
    return response.generations[0].text.strip()

@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Disease Diagnosis Chatbot</title>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                background-image: url('/static/bg.jpg');
                background-size: cover;
                background-position: center;
            }
            .header {
                text-align: center;
                padding: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                font-size: 28px;
                font-weight: bold;
                position: fixed;
                top: 0;
                width: 100%;
            }
            .content {
                margin-top: 100px;
                margin-bottom: 50px;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-grow: 1;
            }
            .chatbot-container {
                background-color: rgba(248, 249, 250, 0.9);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 700px;
                text-align: center;
            }
            .chatbot-question {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
            .chatbot-input {
                margin-top: 10px;
            }
            .button-container {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
            }
            .button-container button {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                background-color: #007BFF;
                color: white;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            .input-field {
                width: 100%;
                padding: 8px;
                margin: 5px 0;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .button-container button.reset {
                background-color: #dc3545;
            }
            .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #007BFF;
            animation: spin 1s ease infinite;
            margin: 0 auto;
            }
            @keyframes spin {
            to { transform: rotate(360deg); }
            }

            #result {
                text-align: left;
                white-space: pre-line;
            }
        </style>
    </head>
    <body>

    <div class="header">
        Disease Diagnosis Chatbot
    </div>

    <div class="content">
        <div class="chatbot-container">
            <div id="question-container">
                <div class="chatbot-question" id="question">Hi! What's your name?</div>
                <div id="questionNumber" class="chatbot-question"></div>
                <div class="chatbot-input" id="input-container">
                    <!-- Input fields will be dynamically inserted here -->
                </div>
            </div>

            <div class="button-container">
                <button id="prevButton" onclick="prevQuestion()" disabled>Previous</button>
                <button class="reset" onclick="resetChat()">Reset</button>
            </div>

            <div id="result" style="display: none;">
                <div class="chatbot-question" id="resultMessage"></div>
                <div id="detailedInfo"></div>
            </div>
        </div>
    </div>

    <div id="loading" style="display: none;">
        <div class="spinner"></div>
        <div>Loading...</div>
    </div>

    <script>
        let step = 0;
        let userData = {};
        let imageUploaded = false;

        const questions = [
            {
                question: "Hi! What's your name?",
                inputType: "text",
                inputId: "name",
                placeholder: "Enter your name"
            },
            {
                question: "Please enter your age:",
                inputType: "number",
                inputId: "age",
                placeholder: "Enter your age"
            },
            {
                question: "Select your gender:",
                inputType: "buttons",
                options: ["Male", "Female"],
                inputId: "gender"
            },
            {
                question: "Do you have a fever?",
                inputType: "buttons",
                options: ["Yes", "No"],
                inputId: "fever"
            },
            {
                question: "Do you have a cough?",
                inputType: "buttons",
                options: ["Yes", "No"],
                inputId: "cough"
            },
            {
                question: "Do you experience fatigue?",
                inputType: "buttons",
                options: ["Yes", "No"],
                inputId: "fatigue"
            },
            {
                question: "Do you have difficulty breathing?",
                inputType: "buttons",
                options: ["Yes", "No"],
                inputId: "breathing"
            },
            {
                question: "What is your blood pressure?",
                inputType: "buttons",
                options: ["Low", "Normal", "High"],
                inputId: "blood_pressure"
            },
            {
                question: "What is your cholesterol level?",
                inputType: "buttons",
                options: ["Low", "Normal", "High"],
                inputId: "cholesterol"
            }
        ];

        const inputContainer = document.getElementById("input-container");
        const questionElement = document.getElementById("question");
        const prevButton = document.getElementById("prevButton");

        function loadQuestion() {
    const currentQuestion = questions[step];
    questionElement.textContent = currentQuestion.question;

    if (currentQuestion.inputType === "text" || currentQuestion.inputType === "number") {
        inputContainer.innerHTML = `
            <input type="${currentQuestion.inputType}" class="input-field" id="${currentQuestion.inputId}" placeholder="${currentQuestion.placeholder}" required>
        `;
        
        const inputElement = document.getElementById(currentQuestion.inputId);

        // Add event listener for "Enter" key to move to the next question
        inputElement.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                nextQuestion();  // Trigger next question on Enter key press
            }
        });

        inputElement.focus();  // Automatically focus the input field
    } else if (currentQuestion.inputType === "buttons") {
        inputContainer.innerHTML = currentQuestion.options
            .map(option => `<button onclick="nextQuestion('${option}')">${option}</button>`)
            .join("");
    }

    prevButton.disabled = step === 0;
}

        function nextQuestion(answer = null) {
    if (answer !== null) {
        const currentQuestion = questions[step];
        userData[currentQuestion.inputId] = answer;
    } else {
        const inputElement = document.getElementById(questions[step].inputId);
        if (inputElement && inputElement.value) {
            userData[inputElement.id] = inputElement.value;
        }
    }

    if (step < questions.length - 1) {
        step++;
        loadQuestion();
    } else {
        // Show loading spinner and hide question content
        document.getElementById("question-container").style.display = "none";
        document.getElementById("loading").style.display = "block";
        
        // Send userData to the backend for disease prediction
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(userData)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner and show result
            document.getElementById("loading").style.display = "none";
            document.getElementById("result").style.display = "block";
            document.getElementById("resultMessage").textContent = data.disease;

            if (data.info) {
                document.getElementById("detailedInfo").textContent = data.info;
            }
        })
        .catch(error => {
            // Handle error case
            document.getElementById("loading").style.display = "none";
            alert('An error occurred. Please try again.');
        });
    }
}


        function prevQuestion() {
            if (step > 0) {
                step--;
                loadQuestion();
            }
        }

        function resetChat() {
            step = 0;
            userData = {};
            document.getElementById("result").style.display = "none";
            document.getElementById("question-container").style.display = "block";
            loadQuestion();
        }

        loadQuestion();  // Load the first question on page load
    </script>

    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    disease = predict_disease(data)

    if disease is None:
        # No disease was predicted, return a message indicating the user is healthy
        return jsonify({'disease': 'You are all okay', 'info': ''})
    
    # Disease was predicted, get detailed info from Cohere
    detailed_info = get_cohere_disease_info(disease)
    return jsonify({'disease': disease, 'info': detailed_info})


if __name__ == '__main__':
    app.run(debug=True)
