from flask import Blueprint, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import yfinance as yf

from Website.stockPrediction import fetch_stock_data, fetch_stock_by_day, fetch_stock_plot, plot_sp500

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():

    symbol = 'AAPL'
    start = '2019-01-01'
    end = '2020-01-01'

    if request.method == 'POST':
        symbol = request.form.get('user_input')
        start = request.form.get('start_date')
        end = request.form.get('end_date')

    stock_data_describe = fetch_stock_data(symbol, start, end)
    stock_data = fetch_stock_by_day(symbol, start, end)
    plot = fetch_stock_plot(symbol, start, end)
    rand_plot = plot_sp500()
    return render_template('index.html', stock_data_describe=stock_data_describe, stock_data=stock_data, start=start,
                           end=end, symbol=symbol, plot=plot)


