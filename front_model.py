import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig, EarlyStoppingCallback
import evaluate
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from customModel import (customBertForSequenceClassification, customRobertaForSequenceClassification,
                         customGPT2ForSequenceClassification, customElectraForSequenceClassification,bigModelCustomRobertaForSequenceClassification,
                         addLayerCustomRobertaForSequenceClassification)
from CustomTraniner import CustomTrainer, CustomTrainer_cross_entropy, CustomTrainer_add_loss, BingModelCustomTrainer
from transformers.configuration_utils import PretrainedConfig
import wandb
import random
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from frontModelCustom import frontModelDataset, data_labels


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    
    best_class = np.argmax(logits, axis=1)
    ret[list(range(len(ret))), best_class] = 1
    return ret

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions_emotion = get_preds_from_logits(logits[0])
    predictions_tempo = get_preds_from_logits(logits[1])
    predictions_genre = get_preds_from_logits(logits[2])
    
    # Get f1 metrics for global scoring. Notice that f1_micro = accuracy
    final_metrics["f1_emotion"] = f1_score(labels[0], predictions_emotion, average="micro")
    
    # Get f1 metrics for causes
    final_metrics["f1_tempo"] = f1_score(labels[1], predictions_tempo, average="micro")
    

    # The global f1_metrics
    final_metrics["f1_genre"] = f1_score(labels[2], predictions_genre, average="micro")

    final_metrics['f1_total'] = (final_metrics["f1_emotion"] + final_metrics["f1_tempo"] + final_metrics["f1_genre"])/3
    
    return final_metrics

data = pd.read_csv('./total_data_no_world.csv') 

emotion , tempo, genre = data_labels('labels.pkl')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)




    BASE_MODEL = 'FacebookAI/roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.num_labels1 = len(emotion)
    config.num_labels2 = len(tempo)
    config.num_labels3 = len(genre)

    model = customRobertaForSequenceClassification.from_pretrained(BASE_MODEL, config= config).to(device)
    


    data2 = data.copy()
    # emotion_data = data2.groupby('emotion').sample(frac=0.05, random_state=42)
    # tempo_data = data2.groupby('tempo(category)').sample(frac=0.05, random_state=42)
    # genre_data = data2.groupby('genre').sample(frac=0.05, random_state=42)
    # index_total = set(emotion_data.index) | set(tempo_data.index) | set(genre_data.index)
    # valid_data = data2.iloc[list(index_total)]
    # train_data = data2.drop(list(index_total)).sample(frac=1, random_state=42)
    data_valid_index = data2.groupby(['emotion','genre','tempo(category)']).sample(frac=0.1, random_state=42).index
    valid_data = data2.iloc[data_valid_index]
    train_data = data2.drop(list(data_valid_index)).sample(frac=1, random_state=42)


    dataset_train = frontModelDataset(train_data, tokenizer =tokenizer)
    dataset_valid = frontModelDataset(valid_data, tokenizer =tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    print(config.num_labels1, config.num_labels2, config.num_labels3)

    wandb.init()

    
    wandb_config = wandb.config
    
    set_seed(wandb_config.seed)

    training_args = TrainingArguments(

    output_dir="my_awesome_model",
    save_steps=300,
    eval_steps = 300, 
    warmup_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    evaluation_strategy='steps',
    load_best_model_at_end = True,
    save_total_limit = 2,
    report_to="wandb",
    metric_for_best_model='f1_total',
    # run_name=BASE_MODEL, 
    )



    trainer = CustomTrainer_add_loss(

    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )


    trainer.train()

if __name__ == "__main__":
    sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "eval/f1_total"},
    "parameters": {
        # "lr": {"max": 0.1, "min": 0.01},
        "seed": {"values": [16,512,50,48,1024,350,700,30,90]},
    },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Final project")

    wandb.agent(sweep_id, function=main, count=9)