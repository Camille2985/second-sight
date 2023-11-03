import os


def get_image_paths_and_labels_new(samples):
    base_image_path = os.path.join("data", "words")
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(base_image_path, partI, partI + "-" + partII, image_name + ".png")
        # if there is an image corresponding to the path listed in the textfile, then append the path name of the image to the ‘paths’ list
        if os.path.isfile(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])
        else:
            paths.append("NO_FILE")
            corrected_samples.append(file_line.split("\n")[0])
    return paths, corrected_samples





def keep_only_existing(img_paths, labels):
    """Given a list of image paths and labels, retain only the image paths that exist and their corresponding labels"""
    img_paths_existing = []
    labels_existing = []

    for i, path in enumerate(img_paths):
        if path != "NO_FILE":
            img_paths_existing.append(path)
            labels_existing.append(labels[i])
    return img_paths_existing, labels_existing


def clean_data(train_samples, validation_samples, test_samples):
    train_img_paths, train_labels = get_image_paths_and_labels_new(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels_new(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels_new(test_samples)

    train_img_paths_existing, train_labels_existing = keep_only_existing(train_img_paths, train_labels)
    validation_img_paths_existing, validation_labels_existing = keep_only_existing(validation_img_paths,
                                                                                   validation_labels)
    test_img_paths_existing, test_labels_existing = keep_only_existing(test_img_paths, test_labels)

    return train_img_paths_existing, train_labels_existing, validation_img_paths_existing, validation_labels_existing, test_img_paths_existing, test_labels_existing
