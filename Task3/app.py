
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# تحميل النموذج
with open('E:/MLCellula_Train/3/Hotel Booking/model (1).pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # الحصول على البيانات من النموذج
    form_data = request.form
 #[['number of adults', 'number of children', 'number of weekend nights',
      # 'number of week nights', 'type of meal', 'room type', 'lead time',
      # 'market segment type', 'P-not-C', 'average price', 'special requests',
      # 'booking status', 'reservation_day', 'reservation_month',
#'reservation_year']]
    # تحويل البيانات إلى DataFrame
    features = pd.DataFrame([[
        float(form_data['number_of_adults']),
        float(form_data['number_of_children']),
        float(form_data['number_of_weekend_nights']),
        float(form_data['number_of_week_nights']),
        int(form_data['type_of_meal']),
        int(form_data['room_type']),
        float(form_data['lead_time']),
        int(form_data['market_segment_type']),
        float(form_data['P-not-C']),  # هذا العمود يجب أن يكون float إذا كان رقميًا
        float(form_data['average_price']),
        int(form_data['special_requests']),
        int(form_data['reservation_day']),
        int(form_data['reservation_month']),
        int(form_data['reservation_year'])
    ]], columns=[
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'type of meal', 'room type', 'lead time',
        'market segment type', 'P-not-C', 'average price', 'special requests',
        'reservation_day', 'reservation_month', 'reservation_year'
    ])
    

    # إجراء التنبؤ
    prediction = model.predict(features)
   # تحويل النتيجة إلى نص مفهوم
    if prediction[0] == 1.0:
        prediction_text = "الحجز متوقع: نعم"
    else:
        prediction_text = "الحجز متوقع: لا"

    # عرض النتيجة
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)