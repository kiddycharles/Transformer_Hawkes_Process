import ccxt
import pandas as pd
from datetime import datetime, timedelta


def fetch_historical_data(exchange, symbol, timeframe, since, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def download_five_year_data(exchange, symbol, timeframe='1d'):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 5)  # Five years ago
    since = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    limit = None  # Fetch all available data

    data = fetch_historical_data(exchange, symbol, timeframe, since, limit)
    return data


if __name__ == "__main__":
    exchange_name = ''  # Example: Binance exchange
    exchange = getattr(ccxt, exchange_name)()

    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BCH/USDT',
               'DOGE/USDT', 'XLM/USDT', 'TRX/USDT', 'ETC/USDT', 'ZEC/USDT',
               'XMR/USDT', 'ADA/USDT']  # Example list of symbols

    print(f"Downloading data for {len(symbols)} cryptocurrencies...")
    all_data = {}
    for symbol in symbols:
        print(f"Downloading {symbol} data...")
        crypto_data = download_five_year_data(exchange, symbol)
        all_data[symbol] = crypto_data

    # Saving data to CSV files
    for symbol, data in all_data.items():
        filename = f"{symbol.replace('/', '_')}_5yr_data.csv"
        data.to_csv(filename, index=False)
        print(f"Data for {symbol} saved to {filename}")
