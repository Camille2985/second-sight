import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_metric
from transformers import TrOCRProcessor
from src.model_training import accuracy_by_letter
import numpy as np


def evaluate(model, dataset, batch_size=8, logger=None):
    """Use this function to evaluate the performance of a model on a dataset according to the eval metrics CER, letter_accuracy, and 
    word_accuracy"""

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    print("Running evaluation...")

    acc_letter = []
    acc_letter_lower = []

    acc_word = 0
    acc_word_lower = 0

    cer_metric = load_metric("cer")
    cer_metric_lower = load_metric("cer")

    for batch in tqdm(test_dataloader):
        
        # generate predictions
        pixel_values = batch["pixel_values"].to(device)
        outputs = model.generate(pixel_values)

        # decode predictions
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)

        # decode labels
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        label_str_lower = [label.lower() for label in label_str]
        pred_str_lower = [pred.lower() for pred in pred_str]

        # cer
        cer_metric.add_batch(predictions=pred_str, references=label_str)
        cer_metric_lower.add_batch(predictions=pred_str_lower, references=label_str_lower)

        # letter accuracy, word accuracy
        for i in range(len(pred_str)):
            acc_letter.append(accuracy_by_letter(pred_str[i], label_str[i]))
            acc_letter_lower.append(accuracy_by_letter(pred_str_lower[i], label_str_lower[i]))
            if pred_str[i] == label_str[i]:
                acc_word += 1
            if pred_str_lower[i] == label_str_lower[i]:
                acc_word_lower += 1
    
    logger.log("CASE SENSITIVE")
    logger.log(f"CER: {cer_metric.compute()}") 
    logger.log(f"Letter accuracy: {np.mean(acc_letter)}")
    logger.log(f"Word accuracy: {acc_word/len(dataset)}")
    logger.log("CASE INSENSITIVE")
    logger.log(f"CER: {cer_metric_lower.compute()}") 
    logger.log(f"Letter accuracy: {np.mean(acc_letter_lower)}")
    logger.log(f"Word accuracy: {acc_word_lower/len(dataset)}")

    return {"cer": cer_metric.compute(), "letter_accuracy": np.mean(acc_letter), "word_accuracy": acc_word/len(dataset)}
