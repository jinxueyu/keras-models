import datetime

__author__ = 'xueyu'


def string_toDatetime(string):
    return datetime.datetime.strptime(string, "%Y%m%d")


def compute_profit(idx):
    profit_date = 1

    if idx < len(trade_date_list):
        today_date_str = trade_date_list[idx]
        last_trade_date_str = trade_date_list[idx-1]
        today_date = string_toDatetime(today_date_str)
        last_trade_date = string_toDatetime(last_trade_date_str)
        profit_date = (today_date - last_trade_date).days

    # trade_date_list
    stocks = [s[idx-1] for s in stock_price_pool]
    profit_rate = stocks[stock_hold]
    # print total_asset, total_asset*profit_date * (profit_rate/100/365), profit_rate
    return total_asset*profit_date * (profit_rate/365)


def trade():
    profit = 0
    trade_count = 0
    global stock_hold, total_asset
    for idx in range(0, len(trade_date_list)):
        if idx > 0:
            profit = compute_profit(idx)
            total_asset += profit

        trade_date = trade_date_list[idx]
        stocks = [s[idx] for s in stock_price_pool]
        stock = 0
        rate = 0
        for stock_idx in range(0, len(stocks)):
            if stocks[stock_idx] > rate:
                stock = stock_idx
                rate = stocks[stock_idx]
        if stock != stock_hold:
            trade_count += 1
        stock_hold = stock

    profit = compute_profit(len(trade_date_list))
    total_asset += profit

    print total_asset/1, trade_count

trade_date_list = []
stock_price_pool = []
stock_hold = None
total_asset = 1


def load_stock_price(stock_file, ratio=1):
    with open('/Users/xueyu/Downloads/'+stock_file) as reader:
        stock_price_list = map(lambda x: float(x[1])*ratio,
                               sorted(map(lambda x: x.rstrip().split('\t'), reader.readlines()),
                                      key=lambda x: x[0],
                                      reverse=False))
        stock_price_pool.append(stock_price_list)


def init_data():
    global trade_date_list, stock_price_list
    with open('/Users/xueyu/Downloads/trade_date.date') as reader:
        trade_date_list = map(lambda x: x.rstrip(), reader.readlines())

    load_stock_price('gc001.price', 0.01)
    load_stock_price('D10003.price')

if __name__ == '__main__':
    init_data()
    trade()
