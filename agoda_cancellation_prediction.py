from pandas import DataFrame
from sklearn.base import BaseEstimator
import agoda_cancellation_estimator
from utils import split_train_test
from agoda_cancellation_estimator import AgodaCancellationEstimator

import numpy as np
import pandas as pd


def calc_canceling_fund(estimated_vacation_time,
                        cancelling_policy_code,
                        original_selling_amount):
    policy_options = cancelling_policy_code.split("_")
    sum = 0
    for option in policy_options:
        if "D" in option:
            if "P" in option:
                charge = int(option[option.find("D") + 1:option.find("P")])
                charge /= 100
                sum += original_selling_amount * charge
            if "N" in option:
                charge = int(option[option.find("D") + 1:option.find("N")])
                charge /= estimated_vacation_time
                sum += original_selling_amount * charge
        elif "P" in option:
            charge = int(option[option.find("D") + 1:option.find("P")])
            charge /= 100
            sum += original_selling_amount * charge
    return sum / len(policy_options)


def data_conversion(full_data: DataFrame):
    country_to_hashcode(full_data)
    accommodation_to_hashcode(full_data)
    payment_type_to_hashcode(full_data)
    dist_checking_checkout = calculate_dates(full_data)
    calculate_canceling_fund_present(full_data, dist_checking_checkout)


def payment_type_to_hashcode(full_data):
    payment_type = set(full_data["original_payment_type"])
    payment_type_dict = {k: v for v, k in enumerate(payment_type)}
    full_data.replace({"original_payment_type": payment_type_dict},
                      inplace=True)
    bool_dict = {True: 0, False: 1}
    full_data.replace({"is_first_booking": bool_dict}, inplace=True)


def accommodation_to_hashcode(full_data):
    accommodation_type = set(full_data["accommadation_type_name"])
    accommodation_type_dict = {k: v for v, k in enumerate(accommodation_type)}
    full_data.replace({"accommadation_type_name": accommodation_type_dict},
                      inplace=True)


def country_to_hashcode(full_data):
    countries_code = set(full_data["origin_country_code"]).union(
        set(full_data["hotel_country_code"]))
    countries_code_dict = {k: v for v, k in enumerate(countries_code)}
    full_data.replace({"origin_country_code": countries_code_dict},
                      inplace=True)
    full_data.replace({"hotel_country_code": countries_code_dict},
                      inplace=True)


def calculate_canceling_fund_present(full_data, dist_checking_checkout):
    cancellation_policy_code = full_data["cancellation_policy_code"]
    original_selling_amount = full_data["original_selling_amount"]
    estimated_vacation_time = pd.to_numeric(dist_checking_checkout.dt.days)
    list_of_tuples = list(zip(estimated_vacation_time, cancellation_policy_code, original_selling_amount, ))
    temp = pd.DataFrame(list_of_tuples, columns=[['time', 'policy', 'cost']])
    res = temp.apply(lambda x: calc_canceling_fund(x['time'], x['policy'], x['cost']), axis=1)
    full_data["avg_cancelling_fund_percent"] = res


def calculate_dates(full_data):
    charge_option_to_hashcode(full_data)
    checking_date = pd.to_datetime(full_data["checkin_date"])
    checkout_date = pd.to_datetime(full_data["checkout_date"])
    booking_date = pd.to_datetime(full_data["booking_datetime"])
    cancel_date = pd.to_datetime(full_data["cancellation_datetime"])
    dist_booking_checking = checking_date - booking_date
    dist_checking_checkout = checkout_date - checking_date
    dist_booking_canceling = cancel_date - booking_date
    full_data["time_from_booking_to_check_in"] = dist_booking_checking
    full_data["estimated_stay_time"] = dist_checking_checkout
    full_data["cancelling_days_from_booking"] = dist_booking_canceling
    return dist_checking_checkout


def charge_option_to_hashcode(full_data):
    charge_option = set(full_data["charge_option"])
    num_charge_option = {value: key for key, value in
                         enumerate(charge_option)}
    full_data.replace({"charge_option": num_charge_option}, inplace=True)


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    insert_new_columns(full_data)
    data_conversion(full_data)
    features = full_data[["time_from_booking_to_check_in",
                          "estimated_stay_time",
                          # "accommadation_type_name",
                          # "charge_option",
                          "guest_is_not_the_customer",
                          # "no_of_adults",
                          # "no_of_children",
                          # "original_selling_amount",
                          "time_from_booking_to_check_in",
                          "original_payment_type",
                          # "no_of_room",
                          "hotel_star_rating",
                          # "origin_country_code",
                          # "hotel_country_code",
                          "is_first_booking",
                          "avg_cancelling_fund_percent"]]
    labels = full_data["cancellation_datetime"]
    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    train_X = train_X.fillna(0)
    train_y = train_y.fillna(0)
    # Fit model over data
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy()
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
