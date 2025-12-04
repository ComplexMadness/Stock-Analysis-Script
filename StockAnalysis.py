from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


# Technical Indicators
def add_indicators(df):
    close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    df['SMA_5'] = close.rolling(5, min_periods=1).mean()
    df['SMA_20'] = close.rolling(20, min_periods=1).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1)))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Volume_SMA'] = volume.rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA'].replace(0, 1)
    df['Momentum'] = close - close.shift(10)

    return df


# Prediction Model
def predict_signals(df):
    signals = []
    for i in range(len(df)):
        score = (1 if df['SMA_5'].iloc[i] > df['SMA_20'].iloc[i] else -1)
        rsi = df['RSI'].iloc[i]
        score += (1 if 30 < rsi < 50 else 0.5 if 50 < rsi < 70 else -1 if rsi > 70 else 0.5)
        score += (1 if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] else -1)
        score += (0.5 if df['Volume_Ratio'].iloc[i] > 1.2 else 0)
        score += (1 if df['Momentum'].iloc[i] > 0 else -1)
        signals.append(1 if score > 0 else 0)

    df['Predicted_Trend'] = ['Up' if s == 1 else 'Down' for s in signals]
    return df


def calculate_accuracy(df):
    df['Actual'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Predicted'] = (df['Predicted_Trend'] == 'Up').astype(int)
    correct = (df['Actual'] == df['Predicted']).sum()
    return correct / len(df.dropna()) if len(df) > 0 else 0.5


# Future Predictions
def predict_future(data, periods, interval):
    last_dt = data.index[-1]
    future_times = []
    current = last_dt

    if interval == '1d':
        for _ in range(periods):
            current += timedelta(days=1)
            while current.weekday() >= 5:
                current += timedelta(days=1)
            future_times.append(current)
    elif interval == '1h':
        count = 0
        while count < periods:
            current += timedelta(hours=1)
            if current.weekday() < 5 and 9 <= current.hour < 16:
                future_times.append(current)
                count += 1
    else:  # 30m
        count = 0
        while count < periods:
            current += timedelta(minutes=30)
            if current.weekday() < 5 and 9 <= current.hour < 16 and current.minute in [0, 30]:
                if not (current.hour == 9 and current.minute == 0):
                    future_times.append(current)
                    count += 1

    future_df = pd.DataFrame(index=pd.DatetimeIndex(future_times))
    last_close = float(data['Close'].iloc[-1])
    last_sma5, last_sma20 = float(data['SMA_5'].iloc[-1]), float(data['SMA_20'].iloc[-1])
    last_rsi, last_macd = float(data['RSI'].iloc[-1]), float(data['MACD'].iloc[-1])
    last_macd_signal = float(data['MACD_Signal'].iloc[-1])

    prices, trends = [], []
    price = last_close
    vol = 0.012 if interval == '1d' else (0.004 if interval == '1h' else 0.002)

    for _ in range(periods):
        score = 0
        recent = data['Predicted_Trend'].iloc[-5:].value_counts()
        score += (1 if 'Up' in recent.index and recent['Up'] > 3 else -1 if 'Down' in recent.index and recent[
            'Down'] > 3 else 0)
        score += (1 if last_sma5 > last_sma20 else -1)
        score += (0.5 if 30 < last_rsi < 70 else -1 if last_rsi > 70 else 1)
        score += (1 if last_macd > last_macd_signal else -1)

        pred = 1 if score > 0 else 0
        trends.append('Up' if pred == 1 else 'Down')
        change = np.random.normal(0, vol)
        price *= (1 + (abs(change) if pred == 1 else -abs(change)))
        prices.append(price)

        last_sma5 = last_sma5 * 0.8 + price * 0.2
        last_sma20 = last_sma20 * 0.95 + price * 0.05
        last_rsi = last_rsi * 0.9 + 50 * 0.1

    future_df['Predicted_Trend'] = trends
    future_df['Projected_Close'] = prices
    return future_df


# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        ticker = request.json['ticker'].upper()
        interval = request.json['interval']
        periods = int(request.json['periods'])

        if interval == '1d':
            years = int(request.json['history'])
            start = datetime.now() - timedelta(days=years * 365)
        else:
            days = int(request.json['history'])
            start = datetime.now() - timedelta(days=days)

        data = yf.download(ticker, start=start, end=datetime.now(), interval=interval, progress=False)
        if data.empty:
            return jsonify({'error': 'No data found for ticker'})

        data = add_indicators(data)
        data = predict_signals(data)
        accuracy = calculate_accuracy(data)
        future = predict_future(data, periods, interval)

        # Current price
        try:
            info = yf.Ticker(ticker).info
            current = info.get("regularMarketPrice", info.get("currentPrice", float(data['Close'].iloc[-1])))
        except:
            current = float(data['Close'].iloc[-1])

        last_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        # Summary
        up_count = (future['Predicted_Trend'] == 'Up').sum()
        down_count = (future['Predicted_Trend'] == 'Down').sum()
        end_price = float(future['Projected_Close'].iloc[-1])
        proj_change = ((end_price - last_close) / last_close * 100)

        # Historical data for chart
        hist_data = {
            'dates': data.index[-100:].strftime('%Y-%m-%d %H:%M').tolist(),
            'prices': data['Close'].iloc[-100:].values.flatten().tolist()
        }

        # Future data for chart
        future_data = {
            'dates': future.index.strftime('%Y-%m-%d %H:%M').tolist(),
            'prices': future['Projected_Close'].values.tolist(),
            'trends': future['Predicted_Trend'].values.tolist()
        }

        # Technical indicators
        last = data.iloc[-1]
        indicators = {
            'rsi': float(last['RSI']),
            'macd': float(last['MACD']),
            'macd_signal': float(last['MACD_Signal']),
            'sma5': float(last['SMA_5']),
            'sma20': float(last['SMA_20'])
        }

        return jsonify({
            'success': True,
            'current_price': current,
            'change': change,
            'change_pct': change_pct,
            'accuracy': accuracy,
            'up_count': int(up_count),
            'down_count': int(down_count),
            'proj_change': proj_change,
            'historical': hist_data,
            'future': future_data,
            'indicators': indicators,
            'data_points': len(data)
        })

    except Exception as e:
        return jsonify({'error': str(e)})


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Stock Trend Analyzer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; margin-bottom: 30px; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-weight: 600; margin-bottom: 5px; color: #555; }
        input, select { padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button { background: #2563eb; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: 600; }
        button:hover { background: #1d4ed8; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #2563eb; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 5px; }
        .metric-value { font-size: 24px; font-weight: 700; color: #333; }
        .metric-delta { font-size: 14px; margin-top: 5px; }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        #chart, #futureChart { margin: 20px 0; }
        .indicators { background: #f8f9fa; padding: 20px; border-radius: 6px; margin: 20px 0; }
        .indicator-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .indicator { padding: 10px; background: white; border-radius: 4px; }
        .indicator-name { font-size: 12px; color: #666; text-transform: uppercase; }
        .indicator-value { font-size: 18px; font-weight: 600; margin-top: 5px; }
        .recommendation { background: #f0f9ff; padding: 20px; border-radius: 6px; border-left: 4px solid #2563eb; margin: 20px 0; }
        .loading { display: none; text-align: center; margin: 20px; }
        .error { background: #fee; color: #c00; padding: 15px; border-radius: 4px; margin: 20px 0; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Trend Analyzer & Predictor</h1>

        <div class="form-grid">
            <div class="form-group">
                <label>Ticker Symbol</label>
                <input type="text" id="ticker" value="AAPL" placeholder="e.g., AAPL">
            </div>
            <div class="form-group">
                <label>Interval</label>
                <select id="interval">
                    <option value="1d">Daily (1d)</option>
                    <option value="1h">Hourly (1h)</option>
                    <option value="30m">30-minute (30m)</option>
                </select>
            </div>
            <div class="form-group" id="historyGroup">
                <label>Years of History</label>
                <input type="number" id="history" value="5" min="1" max="10">
            </div>
            <div class="form-group">
                <label>Forecast Periods</label>
                <input type="number" id="periods" value="30" min="5" max="120">
            </div>
            <div class="form-group" style="justify-content: flex-end;">
                <button onclick="analyze()">Run Analysis</button>
            </div>
        </div>

        <div class="loading" id="loading">Analyzing...</div>
        <div class="error" id="error"></div>

        <div id="results" style="display: none;">
            <div class="metrics" id="metrics"></div>
            <div class="indicators" id="indicators"></div>
            <div id="chart"></div>
            <div id="futureChart"></div>
            <div class="recommendation" id="recommendation"></div>
        </div>
    </div>

    <script>
        document.getElementById('interval').addEventListener('change', function() {
            const histGroup = document.getElementById('historyGroup');
            const histInput = document.getElementById('history');
            const periodsInput = document.getElementById('periods');

            if (this.value === '1d') {
                histGroup.querySelector('label').textContent = 'Years of History';
                histInput.value = '5';
                histInput.max = '10';
                periodsInput.value = '30';
                periodsInput.max = '120';
            } else {
                histGroup.querySelector('label').textContent = 'Days of History';
                histInput.value = this.value === '1h' ? '30' : '7';
                histInput.max = '60';
                periodsInput.value = '20';
                periodsInput.max = '50';
            }
        });

        async function analyze() {
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');

            loading.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ticker: document.getElementById('ticker').value,
                        interval: document.getElementById('interval').value,
                        history: document.getElementById('history').value,
                        periods: document.getElementById('periods').value
                    })
                });

                const data = await response.json();

                if (data.error) {
                    error.textContent = data.error;
                    error.style.display = 'block';
                    return;
                }

                displayResults(data);
                results.style.display = 'block';
            } catch (e) {
                error.textContent = 'Error: ' + e.message;
                error.style.display = 'block';
            } finally {
                loading.style.display = 'none';
            }
        }

        function displayResults(data) {
            // Metrics
            const changeClass = data.change >= 0 ? 'positive' : 'negative';
            const projClass = data.proj_change >= 0 ? 'positive' : 'negative';

            document.getElementById('metrics').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">$${data.current_price.toFixed(2)}</div>
                    <div class="metric-delta ${changeClass}">${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)} (${data.change_pct.toFixed(2)}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">${(data.accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-delta">${data.data_points} data points</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Bullish Periods</div>
                    <div class="metric-value">${data.up_count}</div>
                    <div class="metric-delta positive">${(data.up_count / (data.up_count + data.down_count) * 100).toFixed(0)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Bearish Periods</div>
                    <div class="metric-value">${data.down_count}</div>
                    <div class="metric-delta negative">${(data.down_count / (data.up_count + data.down_count) * 100).toFixed(0)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Projected Change</div>
                    <div class="metric-value ${projClass}">${data.proj_change >= 0 ? '+' : ''}${data.proj_change.toFixed(2)}%</div>
                </div>
            `;

            // Indicators
            const ind = data.indicators;
            const rsiStatus = ind.rsi > 70 ? '(Overbought)' : ind.rsi < 30 ? '(Oversold)' : '(Neutral)';
            const macdStatus = ind.macd > ind.macd_signal ? '(Bullish)' : '(Bearish)';
            const smaStatus = ind.sma5 > ind.sma20 ? '(Golden Cross)' : '(Death Cross)';

            document.getElementById('indicators').innerHTML = `
                <h3 style="margin-bottom: 15px;">Technical Indicators</h3>
                <div class="indicator-grid">
                    <div class="indicator">
                        <div class="indicator-name">RSI</div>
                        <div class="indicator-value">${ind.rsi.toFixed(2)} ${rsiStatus}</div>
                    </div>
                    <div class="indicator">
                        <div class="indicator-name">MACD</div>
                        <div class="indicator-value">${ind.macd.toFixed(4)} ${macdStatus}</div>
                    </div>
                    <div class="indicator">
                        <div class="indicator-name">SMA 5</div>
                        <div class="indicator-value">$${ind.sma5.toFixed(2)}</div>
                    </div>
                    <div class="indicator">
                        <div class="indicator-name">SMA 20</div>
                        <div class="indicator-value">$${ind.sma20.toFixed(2)} ${smaStatus}</div>
                    </div>
                </div>
            `;

            // Charts
            const histTrace = {
                x: data.historical.dates,
                y: data.historical.prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical',
                line: { color: '#2563eb', width: 2 }
            };

            Plotly.newPlot('chart', [histTrace], {
                title: 'Historical Price',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price ($)' },
                height: 400
            });

            const futureTrace = {
                x: data.future.dates,
                y: data.future.prices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Projected',
                line: { color: '#10b981', width: 2, dash: 'dot' },
                marker: { size: 6, color: data.future.trends.map(t => t === 'Up' ? '#10b981' : '#ef4444') }
            };

            Plotly.newPlot('futureChart', [futureTrace], {
                title: 'Projected Price Trend',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price ($)' },
                height: 400
            });

            // Recommendation
            let rec = '';
            if (data.up_count > data.down_count * 1.5) {
                rec = '<strong>STRONG BUY</strong> - High bullish confidence<br>Consider: Long position, call options';
            } else if (data.up_count > data.down_count) {
                rec = '<strong>BUY</strong> - Moderately bullish<br>Consider: Small long position, monitor closely';
            } else if (data.down_count > data.up_count * 1.5) {
                rec = '<strong>STRONG SELL</strong> - High bearish confidence<br>Consider: Short position, put options, or avoid';
            } else if (data.down_count > data.up_count) {
                rec = '<strong>SELL</strong> - Moderately bearish<br>Consider: Reduce position, wait for better entry';
            } else {
                rec = '<strong>NEUTRAL</strong> - Mixed signals<br>Consider: Wait for clearer trend';
            }

            document.getElementById('recommendation').innerHTML = `
                <h3 style="margin-bottom: 10px;">Investment Recommendation</h3>
                <p style="font-size: 16px; line-height: 1.6;">${rec}</p>
                <p style="margin-top: 15px; font-size: 14px; color: #666;">
                    Risk Management: Set stop-loss at 2-3% below entry. Position size: 2-5% of portfolio maximum.
                </p>
            `;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)