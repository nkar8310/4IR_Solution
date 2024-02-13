from flask import Blueprint, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import joblib

# Create a blueprint
xg_blueprint = Blueprint('xg', __name__)

# Define the global variables for model and X_test
model = None
X_test = None

@xg_blueprint.route('/xgboost')
def index():
    return render_template('index2.html')

@xg_blueprint.route('/xgboost/upload', methods=['POST'])
def upload_file():
    global model, X_test
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df['year'] = df.index.year
                df['month'] = df.index.month
                df['day'] = df.index.day
                df['day_of_week'] = df.index.dayofweek

                train_size = int(len(df) * 0.8)
                train, test = df.iloc[:train_size], df.iloc[train_size:]

                X_train, y_train = train.drop(columns='employee_count'), train['employee_count']
                X_test, y_test = test.drop(columns='employee_count'), test['employee_count']

                # Train the model
                model = XGBRegressor(objective='reg:squarederror')
                model.fit(X_train, y_train)

                # Predictions
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                print("Mean Squared Error:", mse)

                # Future predictions
                future_data = pd.DataFrame({'Date': pd.date_range(start='12/31/2023', end='2/28/2024')})
                future = future_data.copy()
                future.set_index('Date', inplace=True)
                future['year'] = future.index.year
                future['month'] = future.index.month
                future['day'] = future.index.day
                future['day_of_week'] = future.index.dayofweek
                future_prediction = model.predict(future)

                # Plotting
                plt.figure(figsize=(18, 6))
                plt.plot(df.index, df['employee_count'].to_list(), label='Actual', marker='o')
                plt.plot(X_test.index, predictions, label='Valid', marker='o')
                plt.plot(future_data['Date'], future_prediction, label='Future', marker='o')
                plt.xlabel('Date')
                plt.ylabel('Employee Count')
                plt.title('Actual vs Predicted Employee Count')
                plt.legend()

                # Save plot as image
                plot_path = 'static/prediction_xgplot.png'
                plt.savefig(plot_path)
                plt.close()
                prediction_df = pd.DataFrame({'Date': future_data['Date'], 'Employee Count': future_prediction})
                prediction_csv_path = 'xgprediction.csv'
                prediction_df.to_csv(prediction_csv_path, index=False)                
                return render_template('index2.html', plot_exists=True, plot_path=plot_path)
            except Exception as e:
                return render_template('index2.html', error=str(e))
    return redirect(url_for('xg.index'))

@xg_blueprint.route('/export', methods=['POST'])
def export():
    global model
    if request.method == 'POST':
        try:
            if model:
                # Save model to file
                model_path = 'static/model.pkl'
                joblib.dump(model, model_path)

                # Provide download link for model file
                return send_file(model_path, as_attachment=True)
            else:
                return render_template('index2.html', error='Model not trained yet')
        except Exception as e:
            return render_template('index2.html', error=str(e))
    return redirect(url_for('xg.index'))


