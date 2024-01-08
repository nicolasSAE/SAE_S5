from flask import Flask, request, jsonify
from flask import Flask, render_template
import joblib

app = Flask(__name__)

# Load the scikit-learn model
model = joblib.load('gb_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Extract features from the data
        features = [
            data['Year'], data['Month'], data['Day'], data['Hour'],
            data['Minute'], data['duration_s'], data['total_counts'],
            data['x_pos_asec'], data['y_pos_asec'], data['radial'],
            data['active_region_ar']
        ]

        # Make predictions with the loaded model
        prediction = model.predict([features])[0]

        # Return the prediction in JSON format
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Handle errors here
        print('Error:', e)
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(port=3000, debug=True)