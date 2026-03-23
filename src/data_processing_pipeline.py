import pandas as pd
import numpy as np

df = pd.read_csv("data/stock_data.csv", header=[0,1], index_col=0)

df.index = pd.to_datetime(df.index)


#rsi function (rsi tells is stock rising too fast or selling too fast)
def add_rsi(data, window=14):

    delta = data['Close'].diff()

    gain = delta.where(delta > 0,0)
    loss = -delta.where(delta < 0,0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain/avg_loss
    data['RSI'] = 100 - (100/(1+rs))

    return data


#macd function (is recent price moving faster than the overall trend)
def add_macd(data):

    exp1 = data['Close'].ewm(span=12, adjust = False).mean()
    exp2 = data['Close'].ewm(span=26, adjust = False).mean()

    data['MACD'] = exp1-exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data


#normal feature enginerring
def process_stock(df, stock_name):
    
    try:
        stock_df = df.xs(stock_name, level=1, axis=1).copy()

        # Remove multi-index column name
        stock_df.columns.name = None

        # Sort by date
        stock_df = stock_df.sort_index()

        # Feature Engineering
        stock_df['Return'] = stock_df['Close'].pct_change()
        stock_df['MA_7'] = stock_df['Close'].rolling(7).mean()
        stock_df['MA_21'] = stock_df['Close'].rolling(21).mean()
        stock_df['Volatility'] = stock_df['Return'].rolling(7).std()
        stock_df['Momentum'] = stock_df['Close'] - stock_df['Close'].shift(7)

        #rsi and macd
        stock_df = add_rsi(stock_df)
        stock_df = add_macd(stock_df)

        # Target
        stock_df['Target'] = stock_df['Close'].shift(-1)

        # Drop NaN
        stock_df = stock_df.dropna()

        return stock_df

    except Exception as e:
        print(f"Error processing {stock_name}: {e}")
        raise e 
    


processed = process_stock(df, "AAPL")
print(processed.head())


#for all stocks

print(df.columns.levels[1])

all_data = []

for stock in df.columns.levels[1]:
    print("Processing:", stock)

    processed = process_stock(df, stock)

    if processed is not None:
        processed['Stock'] = stock
        print("Success:", stock)
        all_data.append(processed)
    else:
        print("Failed:", stock)


final_df = pd.concat(all_data)
final_df = final_df.reset_index()

print(final_df.head())

final_df.to_csv("data/processed_stock_data.csv")

print("Dataset saved succesfully")