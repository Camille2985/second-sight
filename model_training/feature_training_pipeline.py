from src.logging import Logger
from src.model_training import fine_tune
from src.preprocessing import load_data, split_data, save_model, data_preprocessing
from src.torch_processing import IAMDataset
from transformers import TrOCRProcessor
import pickle
import pandas as pd


def pipeline():
    working_locally = True
    data_path = "data/font_rec_new_labels.csv"
    output_path = "output"
    epochs = 10 if working_locally else 30
    gpu = False if working_locally else True
    steps = 25 if working_locally else 200

    # Step 1 - Create Logger
    logger = Logger(output_path)
    logger.log("Pipeline started", False)

    # Step 2 - load the data 
    df = pd.read_csv(data_path)

    # Step 3 - Split the data into train and test sets
    logger.log("Splitting data", False)
    train, validation, test = split_data(df, logger)

    # Step 4 - Data Prep
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
    train_dataset = IAMDataset(df=train,
                               processor=processor,
                               max_target_length=20)
    validation_dataset = IAMDataset(df=validation,
                                    processor=processor,
                                    max_target_length=20)
    test_dataset = IAMDataset(df=test,
                              processor=processor,
                              max_target_length=20)

    # Step 5 - define the model 
    model = pickle.load(open("output/model-large-epoch-20.pkl", 'rb'))

    # Step 6 - Train the model
    logger.log("Training model", False)
    fine_tune(epochs, model, train_dataset, validation_dataset, output_path, gpu, steps)


if __name__ == "__main__":
    pipeline()
