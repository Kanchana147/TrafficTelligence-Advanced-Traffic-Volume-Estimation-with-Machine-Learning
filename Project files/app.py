from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from the form in the same order as the model expects
        features = [
            int(request.form['holiday']),
            float(request.form['temp']),
            float(request.form['rain']),
            float(request.form['snow']),
            int(request.form['weather']),
            int(request.form['year']),
            int(request.form['month']),
            int(request.form['day']),
            int(request.form['hours']),
            int(request.form['minutes']),
            int(request.form['seconds'])
        ]

        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        return render_template("index.html", prediction_text=f"Predicted Traffic Volume: {int(prediction)}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
