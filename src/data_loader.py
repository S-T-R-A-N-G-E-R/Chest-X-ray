import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

def get_dataset(
    csv_path,
    image_root,
    img_size=224,
    batch_size=32,
    shuffle=True,
    augment=False,
    subset=None
):
    """
    Loads NIH Chest X-ray dataset from local or Kaggle environment.

    Args:
        csv_path (str): Path to Data_Entry_2017.csv
        image_root (str): Path to folder containing either `images/` or `images_*` subfolders
        img_size (int): Target image size
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle dataset
        augment (bool): Whether to apply data augmentation
        subset (int): Number of rows to use (debugging)
    Returns:
        dataset (tf.data.Dataset), class_names (list)
    """

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    if subset:
        df = df.sample(n=subset, random_state=42).reset_index(drop=True)

    # --- Process labels ---
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df['Finding Labels'])
    class_names = list(mlb.classes_)

    # --- Map images to full paths ---
    all_image_paths = {}
    for root, dirs, files in os.walk(image_root):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths[fname] = os.path.join(root, fname)

    df['file_path'] = df['Image Index'].map(all_image_paths)
    df = df.dropna(subset=['file_path'])
    label_matrix = label_matrix[df.index]  # Keep only matched labels

    # --- TensorFlow dataset ---
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)  # Grayscale → RGB
        img = tf.image.resize(img, [img_size, img_size])
        img = img / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((df['file_path'].values, label_matrix))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

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
