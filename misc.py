import os

ph = 30
hist = 180
freq = 5
day_len = 1440
cv = 4
seed = 0

nn_models = ["pcLSTM", "LSTM"]

path = "."

data_folder_path = os.path.join("data", "dynavolt")

datasets_subjects_dict = {
    "IDIAB": ["1", "2", "3", "4", "5"],
    "Ohio": ["559", "563", "570", "575", "588", "591"]
}
