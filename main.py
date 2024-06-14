import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import data_preprocess
import transformer

def plot_training_history(t_hist):
    history = t_hist.history
    epochs = range(1, len(history['loss']) + 1)

    # 绘制训练和验证损失
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['loss'], 'b', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

all_predictions = pd.DataFrame()
date = datetime.strptime("2024-01-01", "%Y-%m-%d")

input_shape = (1, 6, 6) # (batch_size, seq_length, n_features)
model = transformer.build_model(
    input_shape,
    head_size=256,
    num_heads=8,
    ff_dim=6,
    num_transformer_blocks=6,
    mlp_units=128,
    dropout=0.1,
    mlp_dropout=0.1,
)

model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-4),
)

all_predictions = pd.DataFrame()  # 存储所有预测值
date = datetime.strptime("2024-01-01", "%Y-%m-%d")
rolling_times = 7  # 滚动训练次数
forward_days = 180  # 滚动训练的时间窗口
backward_days = 20  # 预测的时间窗口
data = []  # 训练与预测数据

for i in range(rolling_times):
    print("rolling: ", i)
    df = data
    train_data = data_preprocess.data_get(df, date - timedelta(days=forward_days), date, train=True)
    train_data_wide, validation_data_wide = data_preprocess.get_train_data(train_data)
    predict_data = data_preprocess.data_get(df, date, date + timedelta(days=backward_days), train=False)
    predict_data_wide = data_preprocess.get_predict_data(predict_data)
    date = date + timedelta(days=backward_days)

    t_hist = model.fit(train_data, validation_data=validation_data_wide, epochs=5, batch_size=1024, verbose=1)
    plot_training_history(t_hist)
    score = model.predict(predict_data)
    score_df = predict_data_wide.on_predict_end(score)
    predictions = predict_data.merge(score_df, on=["date", "instrument"], how="left")
    all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
    all_predictions.to_csv("all_predictions.csv", index=False)