import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os

from src.torch_processing import IAMDataset


def load_data(path):
    words_list = []

    # words = open(f"{words_path}/words.txt", "r").readlines()
    words = open(f"{path}/words.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # remove errored entries
            words_list.append(line)
    print("Total label size:", len(words_list))
    return words_list


def split_data(words_list):
    train_samples, test_samples = train_test_split(words_list,
                                                   test_size=0.1,
                                                   random_state=100)

    validation_samples, test_samples = train_test_split(test_samples,
                                                        test_size=0.5,
                                                        random_state=100)

    print("Total training size:  ", len(train_samples))
    print("Total validation size:", len(validation_samples))
    print("Total test size:      ", len(test_samples))
    print("Total data size:      ", len(words_list))
    print("\nA sample label line:", train_samples[0])

    return train_samples, validation_samples, test_samples



def clean_data(path, samples):
    base_image_path = os.path.join(path, "words")
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

        # if there is an image corresponding to the path listed in the textfile, then append the path name of the image to the 'paths' list
        if os.path.isfile(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])
        else:
            paths.append("NO_FILE")
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels



def data_preprocessing(path, train, validation, test):
    train_img_paths, train_labels = clean_data(path, train)
    validation_img_paths, validation_labels = clean_data(path, validation)
    test_img_paths, test_labels = clean_data(path, test)
    print("Train missing:",
          str(round(100 * len([path for path in train_img_paths if path == "NO_FILE"]) / len(train_img_paths),
                    1)) + "%")
    print("Val missing:  ",
          str(round(100 * len([path for path in validation_img_paths if path == "NO_FILE"]) / len(validation_img_paths),
                    1)) + "%")
    print("Test missing: ",
          str(round(100 * len([path for path in test_img_paths if path == "NO_FILE"]) / len(test_img_paths), 1)) + "%")
    train_labels_cleaned = clean_labels(train_labels)
    validation_labels_cleaned = clean_labels(validation_labels)
    test_labels_cleaned = clean_labels(test_labels)

    # train
    train_df = pd.DataFrame({"file_name": train_img_paths,
                             "text": train_labels_cleaned})

    # validation
    validation_df = pd.DataFrame({"file_name": validation_img_paths,
                                  "text": validation_labels_cleaned})

    # test
    test_df = pd.DataFrame({"file_name": test_img_paths,
                            "text": test_labels_cleaned})

    from transformers import TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    train_dataset = IAMDataset(df=train_df,
                               processor=processor,
                               max_target_length=21)

    validation_dataset = IAMDataset(df=validation_df,
                                    processor=processor,
                                    max_target_length=21)

    test_dataset = IAMDataset(df=test_df,
                              processor=processor,
                              max_target_length=21)

    return train_dataset, validation_dataset, test_dataset

def save_model(model, model_path):
    pickle.dump(model, open(model_path, 'wb'))
