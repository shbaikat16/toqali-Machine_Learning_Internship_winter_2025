from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = pickle.load(open("model.pkl", "rb"))
preprocessing_objects = pickle.load(open("preprocessing.pkl", "rb"))
scaler = preprocessing_objects["scaler"]
label_encoders = preprocessing_objects["label_encoders"]


def preprocess_input(input_data):

    df = pd.DataFrame([input_data])

    # Convert date_of_reservation to datetime and extract reservation month
    df["date_of_reservation"] = pd.to_datetime(
        df["date_of_reservation"], format="%Y-%m-%d", errors="coerce"
    )
    df["reservation_month"] = df["date_of_reservation"].dt.month
    df.drop("date_of_reservation", axis=1, inplace=True)

    # Classify guest type based on number_of_adults and number_of_children
    def classify_guests(row):
        if int(row["number_of_children"]) > 0:
            return "Family"
        elif int(row["number_of_adults"]) == 2 and int(row["number_of_children"]) == 0:
            return "Couples"
        elif int(row["number_of_adults"]) == 1 and int(row["number_of_children"]) == 0:
            return "Singles"
        else:
            return "Groups"

    df["guest_type"] = df.apply(classify_guests, axis=1)

    # Compute total_guests and drop individual guest count columns
    df["total_guests"] = df["number_of_adults"] + df["number_of_children"]
    df.drop(["number_of_adults", "number_of_children"], axis=1, inplace=True)

    # Compute total_nights and drop individual night columns
    df["total_nights"] = df["number_of_weekend_nights"] + df["number_of_week_nights"]
    df.drop(["number_of_weekend_nights", "number_of_week_nights"], axis=1, inplace=True)

    # For consistency with training, convert some columns to string for encoding
    for col in ["reservation_month", "repeated", "car_parking_space"]:
        df[col] = df[col].astype(str)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Standardize numeric columns using the pre-fitted scaler
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Encode categorical columns using saved label encoders
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            df[col] = le.transform(df[col])

    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "date_of_reservation": request.form.get("date_of_reservation"),
            "number_of_adults": int(request.form.get("number_of_adults", "0")),
            "number_of_children": int(request.form.get("number_of_children", "0")),
            "number_of_weekend_nights": int(
                request.form.get("number_of_weekend_nights", "0")
            ),
            "number_of_week_nights": int(
                request.form.get("number_of_week_nights", "0")
            ),
            "room_type": request.form.get("room_type"),
            "lead_time": float(request.form.get("lead_time", "0")),
            "market_segment_type": request.form.get("market_segment_type"),
            "repeated": "1" if request.form.get("repeated") else "0",
            "car_parking_space": "1" if request.form.get("car_parking_space") else "0",
            "p-c": float(request.form.get("pc", "0")),
            "p-not-c": float(request.form.get("p_not_c", "0")),
            "average_price": float(request.form.get("average_price", "0")),
            "special_requests": int(request.form.get("special_requests", "0")),
        }

        processed_df = preprocess_input(input_data)
        final_features = processed_df.values

        prediction = model.predict(final_features)
        pred_prob = model.predict_proba(final_features)
        probability = pred_prob[0][1]
    except Exception as e:
        return jsonify({"prediction": str(e)})

    if prediction[0] == 1:
        output = "Booking will not be cancelled with probability {:.2f}%".format(
            probability * 100
        )
    else:
        output = "Booking will be cancelled with probability {:.2f}%".format(
            (1 - probability) * 100
        )

    return jsonify({"prediction": output})


if __name__ == "__main__":
    app.run(debug=True)
