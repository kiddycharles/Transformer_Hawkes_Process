import pandas as pd
import pickle


def create_crypto_event_sequences(crypto_events):
    single_crypto_sequence = []
    for index, row in crypto_events.iterrows():
        instance = {'time_since_start': row['time_since_start_seconds'],
                    'time_since_last_event': row['time_since_last_event_seconds'], 'type_event': row['mark']}
        single_crypto_sequence.append(instance)
    return single_crypto_sequence


def save_data(data, num_types, filename, type):
    # Create a dictionary to store data and num_types
    saved_data = {type: data, 'dim_process': num_types}

    # Serialize and save the data to a file
    with open(filename, 'wb') as f:
        pickle.dump(saved_data, f)


def split_range_into_categories(min_val, max_val, num_categories=10):
    # Calculate the range
    range_size = max_val - min_val

    # Calculate the size of each category
    category_size = range_size / num_categories

    # Initialize a list to store the category boundaries
    categories = []

    # Iterate to determine the boundaries of each category
    for i in range(num_categories):
        category_min = min_val + i * category_size
        category_max = min_val + (i + 1) * category_size
        categories.append((category_min, category_max))

    # Ensure that the last category ends exactly at max_val
    categories[-1] = (categories[-1][0], max_val)

    return categories


def find_category_index(value, categories):
    for i, (category_min, category_max) in enumerate(categories):
        if category_min <= value <= category_max:
            return i
    return None  # If value is not within any category


def generate_crypto_event_sequences(crypto_list):
    train_event_sequences = []
    test_event_sequences = []
    for crypto_currency in crypto_list:
        df = pd.read_csv('data/crypto/' + crypto_currency + '-USD.csv')
        print(df.shape)
        df['Date'] = pd.to_datetime(df['Date'])
        df['pctc'] = df['Close'].pct_change()
        df = df.fillna(0)
        min_percentage_change = min(df['pctc'])
        max_percentage_change = max(df['pctc'])
        categories = split_range_into_categories(min_percentage_change, max_percentage_change, num_categories=2)
        df['mark'] = df['pctc'].apply(lambda x: find_category_index(x, categories))
        # df['mark'] = (df['Close'] > df['Open']).astype(int)
        # Calculating time since starting time in seconds
        starting_time = df['Date'].iloc[0]
        df['time_since_start'] = (df['Date'] - starting_time)
        # Convert time_since_start column to timedelta datatype
        df['time_since_start'] = pd.to_timedelta(df['time_since_start'])

        # Convert timedelta to seconds
        df['time_since_start_seconds'] = df['time_since_start'].dt.total_seconds() / 60 / 24
        df['time_since_last_event'] = df.groupby('mark')['time_since_start'].diff().fillna(0)
        df['time_since_last_event'] = pd.to_timedelta(df['time_since_last_event'])

        df['time_since_last_event_seconds'] = df['time_since_last_event'].dt.total_seconds() / 60 / 24
        scs = create_crypto_event_sequences(df)
        train_event_sequences.append(scs[:-30])
        test_event_sequences.append(scs[-30:])
        save_data(train_event_sequences, 2, './data/crypto/train_2.pkl', 'train')
        save_data(test_event_sequences, 2, './data/crypto/test_2.pkl', 'test')


if __name__ == '__main__':
    crypto_list = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'USDC', 'USDT', 'XRP']
    generate_crypto_event_sequences(crypto_list)
