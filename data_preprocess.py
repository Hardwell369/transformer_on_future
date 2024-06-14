import hashlib
import math
import os
import types
from datetime import datetime, timedelta

import keras
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split


logger = structlog.get_logger()


class SlidingWindowDataGenerator(keras.utils.PyDataset):
    """时序数据生成器"""
    def __init__(self, data, sequence_length, batch_size):
        self.sequence_length = sequence_length
        self.batch_size = batch_size or 1

        self.feature_cols = [x for x in data.columns if x not in {'date', 'instrument', 'label'}]
        if 'label' in data.columns:
            self.label_col = "label"
        else:
            self.label_col = None

        # 数据预先处理
        data = data.set_index(['date', 'instrument']).unstack(level='instrument').swaplevel(axis=1)
        # 对索引进行排序以确保层级顺序正确
        data.sort_index(axis=1, inplace=True)
        self.data = data

        # 处理 NaN 值，使用填充方法
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')

        if self.label_col is None:
            self.x_data = data.values
            self.y_data = None
        else:
            x_data = data.loc[:, pd.IndexSlice[:, self.feature_cols]]
            x_data.sort_index(axis=1, inplace=True)
            self.x_data = x_data.values

            y_data = data.xs(self.label_col, axis=1, level=1)
            self.y_data = y_data.values

        count = len(self.x_data) - self.sequence_length + 1
        if count < 0:
            raise Exception(f"not enough data for training (data={len(self.x_data)} < sequence_length={self.sequence_length})")
        self.rows = list(range(count))
        self._len = math.ceil(len(self.rows) / self.batch_size)
        self._record_rows = []

    def __len__(self):
        return self._len

    def _extract_data(self, i, all_data, col_count):
        batch = []
        for k in range(i * self.batch_size, min((i + 1) * self.batch_size, len(self.rows))):
            row = self.rows[k]
            self._record_rows.append(row)
            data = all_data[row : row + self.sequence_length]
            data = data.reshape(len(data), data.shape[1] // col_count, col_count)
            # 将数组的轴重新排列
            data = data.transpose(1, 0, 2)
            batch.append(data)

        return np.concatenate(batch, axis=0)

    def __getitem__(self, i):
        x_batch, y_batch = [], []
        for k in range(i * self.batch_size, min((i + 1) * self.batch_size, len(self.rows))):
            row = self.rows[k]
            self._record_rows.append(row)
            # x
            x_data = self.x_data[row : row + self.sequence_length]
            x_data = x_data.reshape(len(x_data), x_data.shape[1] // len(self.feature_cols), len(self.feature_cols))
            # 将数组的轴重新排列
            x_data = x_data.transpose(1, 0, 2)
            x_batch.append(x_data)

            # y
            if self.y_data is not None:
                y_data = self.y_data[row + self.sequence_length - 1]
                y_batch.append(y_data)
            else:
                # fake label
                y_batch.append(np.zeros((len(x_data),)))

        x_batch = np.concatenate(x_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.rows)

    def on_predict_end(self, score):
        print(score)

        dates = self.data.index[self.sequence_length-1:]
        instruments = self.data.columns.levels[0]
        df = pd.DataFrame(
            score.squeeze(axis=1).reshape((len(dates), len(instruments))),
            index=dates,
            columns=instruments,
        ) #.unstack() #.reset_index()
        df = df.melt(ignore_index=False, var_name='instrument', value_name='score').reset_index().sort_values(["date", "instrument"])
        print(df)
        # date, instrument

        return df


def get_train_data(data, input_shape, fit_batch_size):
    logger.info("load trading data ..")
    df = data    
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)
    features = [col for col in df.columns if col not in {"date", "instrument", "label"}]
    logger.info(f"train data shape: {df.shape}, features={len(features)}")

    data_generator = None
    if len(input_shape) >= 3:
        sequence_length = input_shape[1]
        if input_shape[2] != len(features):
            raise ValueError(f"input model.input_shape[2] {input_shape[2]} != features {len(features)}")
        batch_size = fit_batch_size
        data_generator = SlidingWindowDataGenerator(df_train, sequence_length, batch_size)
    
    validation_data = None
    if data_generator is not None:
        sequence_length = input_shape[1]
        batch_size = fit_batch_size
        validation_data = SlidingWindowDataGenerator(df_valid, sequence_length, batch_size)

    return (data_generator, validation_data)


def get_predict_data(data, input_shape):
    df = data
    features = [col for col in df.columns if col not in {"date", "instrument", "label"}]

    data_generator = None
    if len(input_shape) >= 3:
        sequence_length = input_shape[1]
        if input_shape[2] != len(features):
            raise ValueError(f"input model.input[0].shape[2] {input_shape[2]} != features {len(features)}")
        batch_size = 1 # TODO params["predict"]["batch_size"]
        data_generator = SlidingWindowDataGenerator(df, sequence_length, batch_size)
        logger.info(f"predict data shape: {df.shape}")

    return (data_generator)


def data_get(data, start_date, end_date, train=True):
    df = data
    df = df[(df["date"] >= start_date) & (df["date"] < end_date)]
    if train:
        df = df.dropna(subset=["label"])
    return df
