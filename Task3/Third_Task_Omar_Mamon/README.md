# Hotel Reservation Prediction System 🏨

A sophisticated Flask web application leveraging machine learning to predict hotel reservation outcomes with high accuracy.

<p align="center">
    <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGIxdDNobXV2bW9zam1ydmt2aHV3a2ttOGx3MHQ1eDZjcG5rbzJjOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/9yRMxLuRqyQ0x3jJXD/giphy.gif" alt="Hotel Reservation Prediction">
</p>

<p align="center">
    Why did the robot apply for a job at the hotel? Because it wanted to work with reservations!
</p>

## 🌐 Live Demo
Experience it live: [Hotel Reservation Predictor](https://hotelreservationpredictionsystem-production.up.railway.app/)

## 📋 Overview
This intelligent system seamlessly integrates advanced machine learning algorithms with a Flask web application to predict hotel reservation outcomes (successful/cancelled) based on multiple parameters and historical patterns.

## ✨ Key Features
- **Smart Predictions**: Advanced ML models for accurate booking predictions
- **Interactive UI**: Clean, responsive web interface
- **Real-time Results**: Instant prediction feedback
- **Cloud Deployment**: Hosted on Railway for high availability
- **Secure Processing**: Safe handling of user inputs
- **Data Visualization**: Clear presentation of results

## 🚀 Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `python app.py`

## 🧠 Machine Learning Models
The system utilizes a combination of five robust machine learning models:
* **Random Forest**: Ensemble learning with multiple decision trees
* **K-Nearest Neighbors (KNN)**: Pattern recognition based on proximity
* **Logistic Regression**: Linear decision boundaries with probabilistic output
* **XGBoost**: Advanced gradient boosting implementation
* **Support Vector Machine (SVM)**: Optimal hyperplane separation

## 📈 Performance Metrics
Our model demonstrates robust performance across key metrics:

### Classification Report for Random Forest
```
               precision    recall  f1-score   support
    Cancelled     0.85      0.83      0.84      3565
Not Cancelled     0.92      0.93      0.92      7310

     accuracy                         0.90     10875
    macro avg     0.88      0.88      0.88     10875
 weighted avg     0.89      0.90      0.90     10875
```

### Key Statistics for Random Forest
- Overall Accuracy: 90%
- Precision: 89%
- Recall: 90%
- F1-Score: 90%

The model shows particularly strong performance in predicting Not Cancelled reservations, with precision and recall both exceeding 92%.

## 🤝 How to Contribute
We welcome contributions to enhance the Hotel Reservation Prediction System. To contribute, follow these steps:
1. Go to the root repository: [Hotel Reservation Prediction System](https://github.com/Spafic/HotelReservationPredictionSystem/)
2. Fork the repository
3. Create a new branch: `git checkout -b feature-branch`
4. Make your changes and commit them: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature-branch`
6. Submit a pull request

Please ensure your code adheres to our coding standards and includes appropriate tests.