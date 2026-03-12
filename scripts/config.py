from pathlib import Path


NUM_EPOCHS = 20

dataset_dir_name = "data"
train_images_dir_name = "train"
test_images_dir_name = "test"
val_images_dir_name = "valid"
test_cases_dir_name = "test_cases"
raw_csv_name = "raw.csv"
test_csv_name = "test.csv"
train_csv_name = "train.csv"
val_csv_name = "val.csv"
raw_images_dir_name = "images"
cleaned_csv_name = "clean.csv"
train_dataset_name = "train.csv"
val_dataset_name = "val.csv"
test_dataset_name = "test.csv"


base_dir = Path(__file__).resolve().parent.parent



dataset_dir = base_dir / dataset_dir_name
raw_csv_path = dataset_dir / raw_csv_name
raw_images_path = dataset_dir / raw_images_dir_name
train_csv_path = dataset_dir / train_csv_name
test_csv_path = dataset_dir / test_csv_name
val_csv_path = dataset_dir / val_csv_name
train_images_dir = dataset_dir / train_images_dir_name
test_eval_images_dir = dataset_dir / test_images_dir_name
val_images_dir = dataset_dir / val_images_dir_name
test_cases_images_dir = dataset_dir / test_cases_dir_name
clean_csv_path = dataset_dir / cleaned_csv_name
train_clean_csv_path = dataset_dir / train_dataset_name
val_clean_csv_path = dataset_dir / val_dataset_name
test_clean_csv_path = dataset_dir / test_dataset_name




