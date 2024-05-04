import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


def create_crypto_event_sequences(crypto_events):
    single_crypto_sequence = []
    for index, row in crypto_events.iterrows():
        instance = {'time_since_start': row['time_since_start_seconds'],
                    'time_since_last_event': row['time_since_last_event_seconds'], 'type_event': row['mark']}
        single_crypto_sequence.append(instance)
    return single_crypto_sequence


def save_data(data, num_types, filename, type, training_weight=None):
    # Create a dictionary to store data and num_types
    if training_weight is None:
        saved_data = {type: data, 'dim_process': num_types}
    else:
        saved_data = {type: data, 'dim_process': num_types, 'weight': training_weight}

    # Serialize and save the data to a file
    with open(filename, 'wb') as f:
        pickle.dump(saved_data, f)


def split_range_into_categories(min_val, max_val, num_categories=4):
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


def generate_crypto_event_sequences(crypto_list, num_type):
    train_event_sequences = []
    test_event_sequences = []
    for crypto_currency in crypto_list:
        df = pd.read_csv('data/crypto/' + crypto_currency + '-USD.csv')
        print(df.shape)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df['Close'].to_numpy().reshape(-1, 1))
        df['n_close'] = scaled_values

        df['Date'] = pd.to_datetime(df['Date'])
        df['pctc'] = df['Close'].pct_change()
        # Find dates where the absolute percentage change is greater than or equal to 0.1
        significant_changes = df[abs(df['pctc']) >= 0.15]['Date']
        print("Date with high volatility:\n", significant_changes)
        # Specify the date for the vertical line
        significant_changes=['2019-04-02', '2020-03-12', '2020-03-13', '2020-03-19', '2022-06-13']
        for significant_change in significant_changes:
            significant_change = pd.to_datetime(significant_change)
            target_date = pd.to_datetime(significant_change)
        # Draw a vertical red line at the specified date
            plt.axvline(x=target_date, color='r', linestyle='--')

        plt.plot(df['Date'], df['pctc'], label=crypto_currency+' percentage change')
        plt.plot(df['Date'], df['n_close'], label=crypto_currency+' close')
        plt.legend()
        plt.show()
        plt.cla()
        # Get the bin counts
        counts, bins, _ = plt.hist(df['pctc'], bins=num_type)

        # Add text annotations for each bin
        for i in range(len(bins) - 1):
            plt.text((bins[i] + bins[i + 1]) / 2, counts[i], str(int(counts[i])), ha='center', va='bottom')

        # Labeling
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram with Counts')
        plt.show()
        plt.cla()

        df = df.fillna(0)
        min_percentage_change = min(df['pctc'])
        max_percentage_change = max(df['pctc'])
        categories = split_range_into_categories(min_percentage_change, max_percentage_change, num_categories=num_type)
        df['mark'] = df['pctc'].apply(lambda x: find_category_index(x, categories))
        # df['mark'] = (df['Close'] > df['Open']).astype(int)
        # Calculating time since starting time in seconds
        starting_time = df['Date'].iloc[0]
        df['time_since_start'] = (df['Date'] - starting_time)
        # Convert time_since_start column to timedelta datatype
        df['time_since_start'] = pd.to_timedelta(df['time_since_start'])

        # weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
        total_train = df['mark'][:-30].shape[0]
        weights = []
        for i in range(num_type):
            weight_for_class_x = total_train / (df['mark'][:-30].value_counts()[i] * num_type)
            weights.append(weight_for_class_x)

        # Convert timedelta to seconds
        df['time_since_start_seconds'] = df['time_since_start'].dt.total_seconds() / 60 / 24
        df['time_since_last_event'] = df.groupby('mark')['time_since_start'].diff().fillna(0)
        df['time_since_last_event'] = pd.to_timedelta(df['time_since_last_event'])

        df['time_since_last_event_seconds'] = df['time_since_last_event'].dt.total_seconds() / 60 / 24
        scs = create_crypto_event_sequences(df)
        train_event_sequences.append(scs[:-30])
        test_event_sequences.append(scs[-30:])
        # save_data(train_event_sequences, num_type, './data/crypto/train_'+str(num_type)+'_8crypto.pkl', 'train')
        # save_data(test_event_sequences, num_type, './data/crypto/test_'+str(num_type)+'_8crypto.pkl', 'test')


if __name__ == '__main__':
    # crypto_list = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'USDC', 'USDT', 'XRP']
    crypto_list = ['BTC', 'ETH']
    # crypto_list = ['BTC']
    num_type = 8
    generate_crypto_event_sequences(crypto_list, num_type)
