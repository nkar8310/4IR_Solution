from flask import Blueprint, render_template, request, send_file
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Create a blueprint for Facebook Prophet functionalities
fb_blueprint = Blueprint('fb', __name__)

def process_csv(file):
    df = pd.read_csv(file)
    df.rename(columns={'Date': 'ds', 'employee_count': 'y'}, inplace=True)
    return df

def split_train_test(data):
    data['ds'] = pd.to_datetime(data['ds'])
    train = data[data['ds'] <= '2023-11-30']
    valid = data[(data['ds'] > '2023-11-30') & (data['ds'] <= '2023-12-31')]
    return train, valid

@fb_blueprint.route('/export', methods=['GET'])
def export():
    document_name = request.args.get('document_name')
    if not document_name:
        return "Error: Document name not provided"
    try:
        return send_file(document_name + '.csv')
    except Exception as e:
        return str(e)

@fb_blueprint.route('/prophet/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        try:
            df = process_csv(file)
            train_data, valid_data = split_train_test(df)
            model = Prophet()
            model.fit(train_data)
            future = pd.DataFrame({'ds': pd.date_range(start='12/1/2023', end='01/31/2024')})
            prediction = model.predict(future)
            predicted = prediction[['ds', 'yhat']]
            actual = valid_data[['ds', 'y']]
            plt.figure(figsize=(10, 6))
            plt.plot(df['ds'], df['y'], label='Actual', marker='o')
            plt.plot(predicted['ds'], predicted['yhat'], label='Predicted', marker='o')
            plt.xlabel('Date')
            plt.ylabel('Employee Count')
            plt.title('Actual vs Predicted Employee Count in Dec 2023')
            plt.legend()
            plt.savefig('static/prediction_fbplot.png')
            prediction.to_csv('fbprediction.csv')
            return render_template('index.html', error=None, plot_exists=True)
        except Exception as e:
            return render_template('index.html', error=str(e))

@fb_blueprint.route('/prophet', methods=['GET'])
def index():
    return render_template('index.html', error=None, plot_exists=False)
