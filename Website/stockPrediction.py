import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
import base64
from keras.models import load_model


def fetch_stock_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)

        return df.describe().to_html(classes="table table-striped")
    except Exception as e:
        return f"An error occurred: {e}"


def fetch_stock_by_day(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)

        return df.to_html(classes="table table-striped")
    except Exception as e:
        return f"An error occurred: {e}"


def fetch_stock_plot(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)

        # Create a line plot
        plt.plot(df.index, df.Close)
        plt.title(f"{symbol} Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()

        return img_data
    except Exception as e:
        return f"An error occurred: {e}"


def plot_sp500():
    sp500 = yf.Ticker('^GSPC')
    sp500 = sp500.history(period='max')

    sp500.plot.line(y='Close', use_index=True)

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close()
    return img_data


def train_and_test(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)

        data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_test = pd.DataFrame(df['Close'][0:int(len(df)*0.70): int(len(df))])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_train_array = scaler.fit_transform(data_train)

        x_train = []
        y_train = []

        for i in range(100, data_train_array.shape[0]):
            x_train.append(data_train_array[i-100: i])
            y_train.append(data_train_array[i, 0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model = load_model('keras')

        from keras.layers import Dense, Dropout, LSTM
        from keras.models import Sequential

        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))


    except Exception as e:
        return f"An error occurred: {e}"


