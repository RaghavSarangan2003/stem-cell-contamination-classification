import pandas as pd
from sklearn.model_selection import train_test_split
import config

LABEL_COLS = ["bacteria", "healthy", "microplasma"]

def create_class_id(df):
    """
    Convert one-hot labels to single class index
    """
    return df[LABEL_COLS].values.argmax(axis=1)

def stratified_split(df, seed=42):
    """
    Perform stratified train / val / test split
    """
    df = df.copy()
    df["class_id"] = create_class_id(df)

    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["class_id"],
        random_state=seed
    )

    # Split temp into val and test (15% each)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_id"],
        random_state=seed
    )

    return train_df, val_df, test_df

def print_split_stats(name, df):
    print(f"\n{name} split size: {len(df)}")
    print(df["class_id"].value_counts(normalize=True))

if __name__ == "__main__":
    # Load clean combined dataset
    clean_df = pd.read_csv(config.clean_csv_path)

    train_df, val_df, test_df = stratified_split(clean_df)

    # Print stats
    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)

    # Save final CSVs
    train_df.to_csv(config.train_clean_csv_path, index=False)
    val_df.to_csv(config.val_clean_csv_path, index=False)
    test_df.to_csv(config.test_clean_csv_path, index=False)

    print("\nStratified split completed successfully.")
