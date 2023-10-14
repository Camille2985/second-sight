from src.utilities import load_data, split_data, save_model, data_preprocessing



def pipeline():
    # Step 1 - Import the data
    data_path = "model_training/data"
    words_list = load_data(data_path)

    # Step 2 - Split the data into train and test sets
    train, validation, test = split_data(words_list)

    # Step 3 - Data Prep
    train_dataset, validation_dataset, test_dataset = data_preprocessing(data_path, train, validation, test)


    # Step 4 - Create a model


    # Step 5 - Train the model
    tuned_regressor = None

    # Step 7 - Save the model artifacts
    save_model(tuned_regressor, "./model_training/output/model.pkl")

if __name__== "__main__":
    pipeline()
