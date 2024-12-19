from flask import Flask, jsonify
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# Kết nối Firebase
def initialize_firebase():
    # Kiểm tra xem Firebase đã được khởi tạo chưa
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'Link to realtime database from firebase'
        })
    else:
        print("Firebase app is already initialized.")


# Đọc dữ liệu từ file CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data Preview:")
    print(data.head())
    data.dropna(inplace=True)
    data['on_count'] = data['on_count'].astype(int)
    data['total_time_seconds'] = data['total_time_seconds'].astype(float)
    data['max_temperature'] = data['max_temperature'].astype(float)
    data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek
    if 'usage_category' not in data.columns:
        data['usage_category'] = (data['total_time_seconds'] > 7200).astype(int)
    return data

# Chuẩn bị dữ liệu
def prepare_data(data, target_column, features=None):
    if features is None:
        features = ['on_count', 'max_temperature', 'total_time_seconds', 'day_of_week']
    X = data[features]
    y = data[target_column]
    return X, y

# Huấn luyện mô hình phân loại
def train_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Huấn luyện mô hình hồi quy
def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Dự đoán cho 7 ngày tiếp theo
def predict_next_7_days(classification_model, regression_model, data, computers, start_date):
    predictions = {}
    for day_offset in range(7):
        current_date = pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)
        day_of_week = current_date.dayofweek
        daily_predictions = {}

        for computer in computers:
            if data[data['computer'] == computer].empty:
                daily_predictions[computer] = {
                    'usage_level': "Không có dữ liệu",
                    'predicted_time_seconds': 0,
                    'predicted_time_hours': 0
                }
                continue

            row = data[data['computer'] == computer].iloc[-1]
            input_data_classification = pd.DataFrame({
                'on_count': [row['on_count']],
                'max_temperature': [row['max_temperature']],
                'total_time_seconds': [row['total_time_seconds']],
                'day_of_week': [day_of_week]
            })
            input_data_regression = pd.DataFrame({
                'on_count': [row['on_count']],
                'max_temperature': [row['max_temperature']],
                'day_of_week': [day_of_week]
            })

            classification_result = classification_model.predict(input_data_classification)[0]
            usage_level = "Nhiều" if classification_result == 1 else "Ít"
            predicted_time = regression_model.predict(input_data_regression)[0]
            daily_predictions[computer] = {
                'usage_level': usage_level,
                'predicted_time_seconds': predicted_time,
                'predicted_time_hours': predicted_time // 3600
            }

        predictions[current_date.strftime('%Y-%m-%d')] = daily_predictions
    return predictions

# Lấy số lần bật từ Firebase
def fetch_on_count():
    ref = db.reference('ComputerOnCount')
    data = ref.get()

    records = []
    for date, computers in data.items():
        if isinstance(computers, dict):
            for computer, details in computers.items():
                if isinstance(details, dict):
                    records.append({
                        'date': date,
                        'computer': computer,
                        'on_count': details.get('onCount', 0)
                    })
    
    df = pd.DataFrame(records)

    # Đồng bộ định dạng cột
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['computer'] = df['computer'].str.strip()

     # Thay thế giá trị NaN trong cột 'on_count' bằng 0
    df['on_count'] = df['on_count'].fillna(0)

    return df

# Lấy thời gian sử dụng từ Firebase
def fetch_computer_usage_time():
    ref = db.reference('ComputerUsageTime')
    data = ref.get()

    records = []
    for date, computers in data.items():
        for computer, details in computers.items():
            total_time = details.get('totalTime', '00:00:00')
            hours, minutes, seconds = map(int, total_time.split(':'))
            total_time_seconds = hours * 3600 + minutes * 60 + seconds
            records.append({
                'date': date,
                'computer': computer,
                'total_time_seconds': total_time_seconds
            })
    
    df = pd.DataFrame(records)

    # Đồng bộ định dạng cột 'date'
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['computer'] = df['computer'].str.strip()

    return df

# Lấy nhiệt độ từ Firebase và tính nhiệt độ cao nhất mỗi ngày
def fetch_temperature_data():
    ref = db.reference('Temperatures_30')
    data = ref.get()

    records = []
    for timestamp, computers in data.items():
        for computer, details in computers.items():
            records.append({
                'timestamp': timestamp,
                'computer': computer,
                'temperature': details.get('temperature', 0)
            })

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H:%M:%S', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Tách cột ngày từ timestamp
    df['date'] = df['timestamp'].dt.date
    df['computer'] = df['computer'].str.strip()

    # Tính nhiệt độ cao nhất mỗi ngày cho từng máy tính
    daily_max_temp = df.groupby(['date', 'computer']).temperature.max().reset_index()
    daily_max_temp.rename(columns={'temperature': 'max_temperature'}, inplace=True)

    return daily_max_temp

# Gộp dữ liệu và lưu vào một file
def save_all_data_to_file():
    # Lấy dữ liệu từ các hàm
    usage_time_data = fetch_computer_usage_time()
    on_count_data = fetch_on_count()
    daily_temperature = fetch_temperature_data()  # Đã tính max_temperature

    # Gộp dữ liệu
    print("Merging datasets...")
    merged_data = usage_time_data.merge(on_count_data, on=['date', 'computer'], how='left')
    merged_data = merged_data.merge(daily_temperature, on=['date', 'computer'], how='left')

    # Kiểm tra dữ liệu gộp
    print("Merged Data Preview:")
    print(merged_data.head())

    # Lưu dữ liệu gộp vào một file CSV
    merged_data.to_csv('NewAll_data_combined_to_firebase.csv', index=False)
    print("All data has been saved to 'NewAll_data_combined_to_firebase.csv'.")


# Route để huấn luyện lại mô hình và trả kết quả dự đoán
@app.route('/train', methods=['GET'])
def train_and_predict():
    try:
        file_path = "NewAll_data_combined_to_firebase.csv"
        data = load_data(file_path)

        # Chuẩn bị dữ liệu cho mô hình phân loại
        X_classification, y_classification = prepare_data(data, 'usage_category')

        # Chuẩn bị dữ liệu cho mô hình hồi quy
        X_regression, y_regression = prepare_data(data, 'total_time_seconds', ['on_count', 'max_temperature', 'day_of_week'])

        # Huấn luyện mô hình phân loại
        classification_model = train_classification_model(X_classification, y_classification)

        # Huấn luyện mô hình hồi quy
        regression_model = train_regression_model(X_regression, y_regression)

        # Dự đoán cho 7 ngày tiếp theo
        computers = data['computer'].unique()
        start_date = (pd.Timestamp.now().date() + pd.Timedelta(days=1))
        # start_date = pd.Timestamp.now().date()
        predictions = predict_next_7_days(classification_model, regression_model, data, computers, start_date)

        # Hiển thị kết quả dự đoán
        for date, daily_predictions in predictions.items():
            print(f"\nNgày {date}:")
            for computer, result in daily_predictions.items():
                print(f"  {computer}: Mức độ sử dụng: {result['usage_level']}, "
                      f"Thời gian dự đoán: {result['predicted_time_seconds']} giây (~{result['predicted_time_hours']} giờ)")

        # Lưu mô hình
        joblib.dump(classification_model, "MoHinh_PhanLoai_classification_model.pkl")
        joblib.dump(regression_model, "MoHinh_Hoiquy_Regression_model.pkl")
        print("Models saved successfully.")

        return jsonify(predictions), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route ping để kiểm tra trạng thái server
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200


@app.route('/prepare_data', methods=['GET'])
def prepare_data_route():
    try:
        # Gọi hàm để lưu dữ liệu từ Firebase vào file
        initialize_firebase()
        save_all_data_to_file()
        return jsonify({"message": "Data prepared successfully!"}), 200  # Trả về mã 200 khi thành công
    except Exception as e:
        return jsonify({'error': f"Error occurred: {str(e)}"}), 400  # Trả về mã 400 và thông báo lỗi nếu có
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

