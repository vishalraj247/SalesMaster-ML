import pandas as pd

def load_data():
    # Load datasets
    sales_train = pd.read_csv('data/raw/sales_train.csv')
    sales_test = pd.read_csv('data/raw/sales_test.csv')
    calendar = pd.read_csv('data/raw/calendar.csv')
    sell_prices = pd.read_csv('data/raw/items_weekly_sell_prices.csv')
    calendar_events = pd.read_csv('data/raw/calendar_events.csv')

    return sales_train, sales_test, calendar, sell_prices, calendar_events

def display_head(sales_train, sales_test, calendar, sell_prices, calendar_events):
    # Display the first few rows of each dataset.
    train_head = sales_train.head()
    test_head = sales_test.head()
    calendar_head = calendar.head()
    sell_prices_head = sell_prices.head()
    calendar_events_head = calendar_events.head()
    
    return train_head, test_head, calendar_head, sell_prices_head, calendar_events_head

def load_data_prediction():
    # Load datasets
    calendar = pd.read_csv('data/raw/calendar.csv')
    calendar_events = pd.read_csv('data/raw/calendar_events.csv')

    return calendar, calendar_events