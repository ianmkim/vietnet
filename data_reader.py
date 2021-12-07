import glob, os
import numpy as np
import pandas as pd
from pprint import pprint

import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reformat_date(date_str):
    sds = date_str.split("/")
    return sds[0].zfill(2) + "/" + sds[2]

index_returns = pd.read_csv("data/vnfin/market_activity/total_return_index_monthly.csv", index_col=0)
index_returns = index_returns.rename(index={index:reformat_date(index) for index, _ in index_returns["VNINDEX Index"].iteritems()})


def quarters(month):
    quarts = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
    }
    return quarts[int(month)]

def parse_data_into_dict():
    os.chdir("data/vnfin")
    #returns = pd.read_csv("market_activity/total_return_index_monthly.csv", index_col=0)

    combined = {}

    for col in index_returns.columns:
        if "Equity" in col:
            combined[col] = {}

    for comp in combined:
        combined[comp]["total_return_index_monthly"] = index_returns[comp]

    for file in glob.glob("monthly/*.csv"):
        data_name = file.replace(".csv", "")
        feature = pd.read_csv(file, index_col=0)
        feature = feature.rename(index={index:reformat_date(index) for index, _ in feature["VNINDEX Index"].iteritems()})
        for comp in combined:
            combined[comp][data_name] = feature[comp]

    quarterlies_glob = glob.glob("quarterly/*.csv")
    for glob_indx, file in enumerate(quarterlies_glob):
        data_name = file.replace(".csv", "")
        feature = pd.read_csv(file, index_col=0)
        total_companies = len(combined)
        for i, comp in enumerate(combined):
            print(f"Total percent: {(glob_indx/len(quarterlies_glob))*100:0.2f}% - Task percent: {(i/total_companies)*100:.02f}%")
            dates = [[int(num) for num in date.split("/")] for date, _ in combined[comp]["total_return_index_monthly"].iteritems()]
            values = []
            for date in dates:
                for q_date, value in feature[comp].iteritems():
                    split_date = [int(split_q_date) for split_q_date in q_date.split("/")]
                    if date[1] == split_date[2] and quarters(date[0]) == quarters(split_date[0]): # if the year is the same and the quarter is the same
                        values.append(value)
            assert(len(dates) == len(values))
            dates = [str(date[0]).zfill(2) + "/" + str(date[1]) for date in dates]
            new_feature_series = pd.Series(values, index=dates)
            combined[comp][data_name] = new_feature_series

    return combined

def load_data(data_path="data/combined.pickle"):
    from os.path import exists
    if exists(data_path):
        with open(data_path, "rb") as handle:
            data = pickle.load(handle)
    else:
        data = parse_data_into_dict()
        os.chdir("../../")
        with open(data_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def load_features(data, forced_reconstruct=False, features_path="data/features.pickle", labels_path="data/labels.pickle"):
    from os.path import exists
    labels = None
    features = None
    if not exists(features_path) or not exists(labels_path) or forced_reconstruct:
        construct_features(data)
    with open(features_path, "rb") as handle:
        features = pickle.load(handle)
    with open(labels_path, "rb") as handle:
        labels = pickle.load(handle)

    return features, labels

def next_month(date, month=1):
    dt = datetime.strptime(date, "%m/%Y")
    dt += relativedelta(months=month)
    return dt.strftime("%m/%Y")

def date_equals(date1, date2):
    ds1 = date1.split("/")
    ds2 = date2.split("/")
    return ds1[0].zfill(2) == ds2[0].zfill(2) and ds1[2] == ds2[2]

def stock_to_index(stock):
    finder = {
        "VH Equity": "VHINDEX Index",
        "VM Equity": "VNINDEX Index"
    }
    for index_name in finder:
        if index_name in stock:
            return finder[index_name]

def get_beg_end_price(column_index, begin_date, end_date):
    return index_returns[column_index][begin_date], index_returns[column_index][end_date]

    begin_market_price = -1
    end_market_price = -1
    for market_dates, market_performance in index_returns[column_index].iteritems():
        if date_equals(market_dates, begin_date) and not np.isnan(market_performance):
            begin_market_price = market_performance
        if date_equals(market_dates, end_date) and not np.isnan(market_performance):
            end_market_price = market_performance

    return begin_market_price, end_market_price


def inspect_nans_in_data(data):
    nan_num_dict = {}
    for progress, comp in enumerate(data):
        for column in data[comp]:
            if progress == 0:
                nan_num_dict[column] = 0
            start_index = -1
            for i, row in data[comp][column].iteritems():
                if not np.isnan(row):
                    start_index = i
            nan_num_dict[column] += data[comp][column][start_index:].isna().sum()

    pprint(nan_num_dict)

def construct_features(data, lookback_period=12, prediction_period=3):
    features = []
    labels = []

    for progress, comp in enumerate(data):
        dates_to_collect = []
        for date, returns in data[comp]["total_return_index_monthly"][:-lookback_period-prediction_period].iteritems():
            if not np.isnan(returns):
                nan_in_dates = False
                dates_inner = [date]
                next_date = date
                for _ in range(lookback_period-1):
                    next_date = next_month(next_date)
                    dates_inner.append(next_date)
                dates_to_collect.append(dates_inner)

        for date_range in dates_to_collect:
            feature_for_range = []
            for date in date_range:
                feature_row = []
                for feature_set in data[comp].values():
                    current_data = feature_set[date]
                    if current_data is not None and not np.isnan(current_data):
                        feature_row.append(current_data)
                if len(feature_row) == len(data[comp]):
                    feature_for_range.append(feature_row)

            if len(feature_for_range) == len(date_range):
                features.append(feature_for_range)

                begin_date = date_range[-1]
                end_date = next_month(date_range[-1], month=prediction_period)
                begin_market_price, end_market_price = get_beg_end_price(stock_to_index(comp), begin_date, end_date)
                begin_stock_price, end_stock_price = get_beg_end_price(comp, begin_date, end_date)
                assert(begin_market_price != -1 and end_market_price != -1 and begin_stock_price != -1 and end_stock_price != -1)

                market_growth = (end_market_price - begin_market_price) / begin_market_price
                stock_growth = (end_stock_price - begin_stock_price) / begin_stock_price

                labels.append(1 if stock_growth >= market_growth else 0)

        print(f"{len(features)} features collected - {(progress/len(data))*100:.02f}%")

    print(f"{len(features)} features collected for {len(labels)} labels")
    print("Saving...")

    with open("data/features.pickle", "wb") as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _construct_features(data, lookback_period=12, prediction_period=3):
    features = []
    labels = []
    for progress, comp in enumerate(data):
        feature_dates = []
        for date, returns in data[comp]["total_return_index_monthly"][:-lookback_period-prediction_period].iteritems():
            if not np.isnan(returns):
                inner_feature_dates = [date]
                next_date = date
                for _ in range(lookback_period-1):
                    next_date = next_month(next_date)
                    inner_feature_dates.append(next_date)
                feature_dates.append(inner_feature_dates)

        for date_range in feature_dates:
            feature = []
            for date in date_range:
                feature_row = []
                for feature_set in data[comp].values():
                    current_data = None
                    for i, r in feature_set.iteritems():
                        if date_equals(i, date):
                            current_data = r
                            break
                    if current_data is not None and not np.isnan(current_data):
                        feature_row.append(current_data)
                print(len(data[comp]))
                if len(feature_row) == len(data[comp]):
                    feature.append(feature_row)

            print(len(feature), len(date_range))
            if len(feature) == len(date_range):
                features.append(feature)

                begin_date = date_range[-1]
                end_date = next_month(date_range[-1], month=prediction_period)
                begin_market_price, end_market_price = get_beg_end_price(stock_to_index(comp), begin_date, end_date)
                begin_stock_price, end_stock_price = get_beg_end_price(comp, begin_date, end_date)
                assert(begin_market_price != -1 and end_market_price != -1 and begin_stock_price != -1 and end_stock_price != -1)

                market_growth = (end_market_price - begin_market_price) / begin_market_price
                stock_growth = (end_stock_price - begin_stock_price) / begin_stock_price

                labels.append(1 if stock_growth >= market_growth else 0)

        print(f"{len(features)} features collected - {(progress/len(data))*100:.02f}%")

    print(f"{len(features)} features collected for {len(labels)} labels")
    print("Saving...")

    with open("data/features.pickle", "wb") as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def features_to_samples(features, labels):
    scaler = StandardScaler()
    flattened_features = []
    features = np.array(features)
    for feature, label in zip(features, labels):
        feature = np.array(feature)
        flattened_feature = feature.flatten()
        assert len(flattened_feature) == feature.shape[0] * feature.shape[1]
        flattened_features.append(flattened_feature)

    scaled_features = scaler.fit_transform(flattened_features).reshape(features.shape[0], features.shape[1], features.shape[2])

    samples = []
    for feature, label in zip(features, labels):
        tup = (torch.Tensor(np.array([feature])), label)
        samples.append(tup)

    training_set, testing_set = train_test_split(samples, test_size=.20, shuffle=True)
    return training_set, testing_set


def check_training_set_balance(training_set, testing_set):
    def check_balance(set):
        num_dict = {
            0: 0,
            1: 0,
        }
        for sample in set:
            num_dict[sample[1]] += 1
        return num_dict

    return check_balance(training_set), check_balance(testing_set)

def load_train_test_sets():
    data = load_data()
    features, labels = load_features(data, forced_reconstruct=False)
    training_set, testing_set = features_to_samples(features, labels)
    return training_set, testing_set

if __name__ == "__main__":
    data = load_data()
    features, labels = load_features(data, forced_reconstruct=False)
    training_set, testing_set = features_to_samples(features, labels)

    pprint(check_training_set_balance(training_set, testing_set))
