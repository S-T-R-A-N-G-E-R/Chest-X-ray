import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


def get_dataset(
    csv_path,
    image_dir,
    img_size=224,
    batch_size=32,
    shuffle=True,
    augment=False,
    subset=None
):
    """
    Loads NIH Chest X-ray dataset as a tf.data.Dataset.
    
    Args:
        csv_path (str): Path to Data_Entry_2017.csv
        image_dir (str): Path to images folder
        img_size (int): Target image size (square)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle dataset
        augment (bool): Whether to apply data augmentation
        subset (int): Number of rows to use (for debugging)

    Returns:
        dataset (tf.data.Dataset), class_names (list)
    """

    # --- Read CSV ---
    df = pd.read_csv(csv_path)

    if subset:
        df = df.sample(n=subset, random_state=42).reset_index(drop=True)

    # --- Process labels ---
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df['Finding Labels'])
    class_names = list(mlb.classes_)

    # Drop original Finding Labels to avoid confusion
    df = df.drop(columns=['Finding Labels'])

    # --- Add file paths ---
    df['file_path'] = df['Image Index'].apply(lambda x: os.path.join(image_dir, x))

    # --- Build TensorFlow dataset ---
    paths = df['file_path'].values
    labels = label_matrix

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = img / 255.0
        return img, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Optional Augmentation ---
    if augment:
        def augment_image(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            return img, label
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_names
