def get_dataset(
    csv_path,
    image_root,
    img_size=224,
    batch_size=32,
    shuffle=True,
    augment=False,
    subset=None
):
    import os
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import MultiLabelBinarizer

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    if subset:
        df = df.sample(n=subset, random_state=42).reset_index(drop=True)

    # --- Map filenames to full paths ---
    all_image_paths = {}
    needed_files = set(df['Image Index'].values)
    for folder in os.listdir(image_root):
        if folder.startswith("images_"):
            folder_path = os.path.join(image_root, folder)
            for fname in os.listdir(folder_path):
                if fname in needed_files:
                    all_image_paths[fname] = os.path.join(folder_path, fname)

    df['file_path'] = df['Image Index'].map(all_image_paths)

    # --- Drop rows without images ---
    df = df.dropna(subset=['file_path']).reset_index(drop=True)

    # --- Process labels AFTER dropping NaNs ---
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df['Finding Labels'])
    class_names = list(mlb.classes_)

    # --- TensorFlow dataset ---
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = img / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((df['file_path'].astype(str).values, label_matrix))
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
