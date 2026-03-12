import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config
from PIL import Image


def get_image_resolution(filename):
    try:
        with Image.open(config.raw_images_path / filename) as img:
            return img.width, img.height
    except:
        return None, None

def add_resolution_columns(df):
    df[["width", "height"]] = df["filename"].apply(
        lambda x: pd.Series(get_image_resolution(x))
    )
    return df


def visualize_before_preprocessing(raw_dataframe):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_cols = ["bacteria", "healthy", "microplasma"]

    # Class distribution
    class_counts = raw_dataframe[label_cols].sum().reset_index()
    class_counts.columns = ["Class", "Count"]
    sns.barplot(data=class_counts, x="Class", y="Count", ax=axes[0, 0])
    axes[0, 0].set_title("Class Distribution (Before Preprocessing)")

    # Label sum distribution
    label_sum = raw_dataframe[label_cols].sum(axis=1)
    sns.countplot(x=label_sum, ax=axes[0, 1])
    axes[0, 1].set_title("Label Sum Distribution (Before Preprocessing)")
    axes[0, 1].set_xlabel("Number of Active Labels")
    axes[0, 1].set_ylabel("Number of Images")

    # Class-wise presence
    long_df = raw_dataframe[label_cols].melt(var_name="Class", value_name="Presence")
    long_df["Presence"] = long_df["Presence"].map({0: "Not Present", 1: "Present"})

    sns.countplot(data=long_df, x="Presence", hue="Class", ax=axes[1, 0])
    axes[1, 0].set_title("Class Label Presence per Image (Before Preprocessing)")

    # Resolution scatter
    sns.scatterplot(data=raw_dataframe, x="width", y="height", alpha=0.6, ax=axes[1, 1])
    axes[1, 1].set_title("Image Resolution Distribution (Before Preprocessing)")

    plt.tight_layout()
    plt.show()




def remove_invalid_images(raw_dataframe):
    label_cols = ["bacteria", "healthy", "microplasma"]

    clean_df = raw_dataframe[raw_dataframe[label_cols].sum(axis=1) == 1].copy()
    clean_df = clean_df.reset_index(drop=True)

    print("Before preprocessing:", len(raw_dataframe))
    print("After preprocessing:", len(clean_df))
    clean_df.to_csv(config.clean_csv_path, index=False)
    return clean_df


def visualize_after_preprocessing(clean_dataframe):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_cols = ["bacteria", "healthy", "microplasma"]

    # Class distribution
    class_counts = clean_dataframe[label_cols].sum().reset_index()
    class_counts.columns = ["Class", "Count"]
    sns.barplot(data=class_counts, x="Class", y="Count", ax=axes[0, 0])
    axes[0, 0].set_title("Class Distribution (After Preprocessing)")

    # Label sum distribution
    label_sum = clean_dataframe[label_cols].sum(axis=1)
    sns.countplot(x=label_sum, ax=axes[0, 1])
    axes[0, 1].set_title("Label Sum Distribution (After Preprocessing)")

    # Class presence
    long_df = clean_dataframe[label_cols].melt(var_name="Class", value_name="Presence")
    long_df["Presence"] = long_df["Presence"].map({0: "Not Present", 1: "Present"})

    sns.countplot(data=long_df, x="Presence", hue="Class", ax=axes[1, 0])
    axes[1, 0].set_title("Class Label Presence per Image (After Preprocessing)")

    # Resolution scatter
    sns.scatterplot(data=clean_dataframe, x="width", y="height", alpha=0.6, ax=axes[1, 1])
    axes[1, 1].set_title("Image Resolution Distribution (After Preprocessing)")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    raw_df = pd.read_csv(config.raw_csv_path)

    raw_df = add_resolution_columns(raw_df)
    visualize_before_preprocessing(raw_df)

    clean_df = remove_invalid_images(raw_df)
    clean_df = add_resolution_columns(clean_df)

    visualize_after_preprocessing(clean_df)

