from datasets import load_metric
import numpy as np
import os
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    TrOCRProcessor, TrainerCallback, default_data_collator
import torch

from src.preprocessing import save_model
from src.logging import Logger

EPOCH = 1


def fine_tune(epochs, train_dataset, validation_dataset, output_path, gpu=False, steps=2 ):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=gpu,
        output_dir=output_path,
        logging_steps=2,
        save_steps=1000,
        eval_steps=steps,
        num_train_epochs=epochs
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=eval_metrics,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        callbacks=[SaveEvaluationResultsCallback()],
    )
    trainer.train()


def accuracy_by_letter(pred, actual):
    max_length = max(len(pred), len(actual))
    min_length = min(len(pred), len(actual))
    letters_correct = 0
    for i in range(min_length):
        if pred[i] == actual[i]:
            letters_correct += 1
    return letters_correct / max_length


def eval_metrics(pred):
    logger = Logger("output")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    global EPOCH
    logger.log(f"EPOCH: {EPOCH}")
    EPOCH += 1

    # case insensitive
    label_str_lower = [label.lower() for label in label_str]
    pred_str_lower = [pred.lower() for pred in pred_str]

    # cer
    cer_metric = load_metric("cer")
    cer_metric_lower = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    cer_lower = cer_metric_lower.compute(predictions=pred_str_lower, references=label_str_lower)

    # letter accuracy, word accuracy
    acc_letter = []
    acc_letter_lower = []
    acc_word = 0
    acc_word_lower = 0
    for i in range(len(pred_str)):
        acc_letter.append(accuracy_by_letter(pred_str[i], label_str[i]))
        acc_letter_lower.append(accuracy_by_letter(pred_str_lower[i], label_str_lower[i]))
        if pred_str[i] == label_str[i]:
            acc_word += 1
        if pred_str_lower[i] == label_str_lower[i]:
            acc_word_lower += 1

    logger.log("CASE SENSITIVE")
    logger.log(f"CER: {cer}")
    logger.log(f"Letter accuracy: {np.mean(acc_letter)}")
    logger.log(f"Word accuracy: {acc_word / len(label_str)}")
    logger.log("CASE INSENSITIVE")
    logger.log(f"CER: {cer_lower}")
    logger.log(f"Letter accuracy: {np.mean(acc_letter_lower)}")
    logger.log(f"Word accuracy: {acc_word_lower / len(label_str)}")

    return {"cer": cer, "letter_accuracy": np.mean(acc_letter), "word_accuracy": acc_word / len(label_str), 
            "cer_lower": cer_lower, "letter_accuracy_lower": np.mean(acc_letter_lower), "word_accuracy_lower": acc_word_lower / len(label_str)}


class SaveEvaluationResultsCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, model, tokenizer, data_collator=None, compute_metrics=None, **kwargs):
        save_model(model, f"./output/model-epoch-{EPOCH}.pkl")
