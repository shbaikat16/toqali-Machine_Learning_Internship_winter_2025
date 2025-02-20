from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained Random Forest model
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Process base inputs
        lead_time = int(data.get('lead_time', 0))
        avg_price = float(data.get('avg_price', 0))
        special_requests = int(data.get('special_requests', 0))
        
        # Process dates
        reservation_year = int(data.get('reservation_year', 2023))
        reservation_month = int(data.get('reservation_month', 1))
        reservation_day = int(data.get('reservation_day', 1))
        
        # Calculate is_weekday based on the date
        date = datetime(reservation_year, reservation_month, reservation_day)
        is_weekday = 1 if date.weekday() < 5 else 0
        
        # Process split member and night inputs
        num_adults = int(data.get('num_adults', 1))
        num_children = int(data.get('num_children', 0))
        total_members = num_adults + num_children
        
        num_weekend_nights = int(data.get('num_weekend_nights', 0))
        num_week_nights = int(data.get('num_week_nights', 0))
        total_nights = num_weekend_nights + num_week_nights
        
        # Process boolean inputs
        old_customer = 1 if data.get('old_customer') == 'on' else 0
        
        # Process cancellation history
        prev_canceled = int(data.get('prev_canceled', 0))
        prev_not_canceled = int(data.get('prev_not_canceled', 0))
        percent_cancel = prev_canceled / (prev_canceled + prev_not_canceled + 1e-10)
        
        # Process room type and car parking
        room_type_mapping = {
            'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 
            'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7
        }
        room_type = room_type_mapping.get(data.get('room_type', 'Room_Type 1'), 1)
        car_parking_space = int(data.get('car_parking_space', 0))
        
        # Process market segment
        market_segment = data.get('market_segment', 'Online')
        market_features = {
            'market segment type_Online': 1 if market_segment == 'Online' else 0,
            'market segment type_Offline': 1 if market_segment == 'Offline' else 0,
            'market segment type_Corporate': 1 if market_segment == 'Corporate' else 0,
        }
        
        # Process meal type
        meal_type = data.get('meal_type', 'Meal Plan 1')
        meal_features = {
            'type of meal_Meal Plan 1': 1 if meal_type == 'Meal Plan 1' else 0,
            'type of meal_Meal Plan 2': 1 if meal_type == 'Meal Plan 2' else 0,
            'type of meal_Not Selected': 1 if meal_type == 'Not Selected' else 0
        }
        
        # Create feature array in the correct order
        feature_names = [
            'lead time', 'average price', 'reservation_year', 'reservation_month', 
            'reservation_day', 'special requests', 'percent_cancel', 'is_weekday',
            'total_members', 'total_nights', 'Old Customer', 'Previously Canceled',
            'Previously not Canceled', 'room type', 'car parking space',
            'market segment type_Online', 'market segment type_Offline', 
            'market segment type_Corporate', 'type of meal_Meal Plan 1',
            'type of meal_Meal Plan 2', 'type of meal_Not Selected'
        ]
        
        # Create the features dictionary with all values
        features_dict = {
            'lead time': lead_time,
            'average price': avg_price,
            'reservation_year': reservation_year,
            'reservation_month': reservation_month,
            'reservation_day': reservation_day,
            'special requests': special_requests,
            'percent_cancel': percent_cancel,
            'is_weekday': is_weekday,
            'total_members': total_members,
            'total_nights': total_nights,
            'Old Customer': old_customer,
            'Previously Canceled': prev_canceled,
            'Previously not Canceled': prev_not_canceled,
            'room type': room_type,
            'car parking space': car_parking_space,
            'market segment type_Online': market_features['market segment type_Online'],
            'market segment type_Offline': market_features['market segment type_Offline'],
            'market segment type_Corporate': market_features['market segment type_Corporate'],
            'type of meal_Meal Plan 1': meal_features['type of meal_Meal Plan 1'],
            'type of meal_Meal Plan 2': meal_features['type of meal_Meal Plan 2'],
            'type of meal_Not Selected': meal_features['type of meal_Not Selected']
        }
        
        # Create feature array in the correct order
        features = [features_dict[name] for name in feature_names]
        
        # Reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Convert prediction to text
        result = "Not Canceled" if prediction else "Canceled"
        
        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
