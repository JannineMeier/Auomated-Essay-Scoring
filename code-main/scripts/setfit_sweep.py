from dotenv import load_dotenv

load_dotenv()
import os
import huggingface_hub
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback
import wandb
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import (
    CosineSimilarityLoss,
    ContrastiveLoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
)
import torch
from sklearn.metrics import cohen_kappa_score
import numpy as np
import re
from collections import Counter
import shutil
from huggingface_hub import HfApi, Repository, ModelCard

# Clear memory for all GPUs before model assignment
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Set environment variables
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Load dataset from Hugging Face hub
huggingface_username = "HSLU-AICOMP-LearningAgencyLab"
competition = "learning-agency-lab-automated-essay-scoring-2"
our_model_name = "automated-essay-scoring-setfit"

wandb_project = "HSLU-AICOMP-LearningAgencyLab"
wandb_entity = "Leo1212"

max_words = 4096
best_qwk_score = -1
best_model_dir = None
best_hyperparameters = None

huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
wandb.login(key=os.getenv("WANDB_API_TOKEN"))

os.environ["WANDB_PROJECT"] = wandb_project

# Load the dataset from Hugging Face
dataset = load_dataset(f"{huggingface_username}/{competition}")


def count_words(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return len(words)


def truncate_text(text, max_words=384):
    words = text.split()
    return " ".join(words[:max_words])


def subsample_dataset(
    dataset, split="train", score_column="score", num_per_score=15, max_words=384
):
    reduced_dataset_list = []

    for score in range(1, 7):
        filtered = dataset[split].filter(lambda x: x[score_column] == score)
        filtered = filtered.map(
            lambda x: {
                "text": (
                    truncate_text(x["full_text"], max_words)
                    if count_words(x["full_text"]) > max_words
                    else x["full_text"]
                )
            }
        )

        if len(filtered) > 0:
            sample_count = min(len(filtered), num_per_score)
            reduced_dataset_list.append(
                filtered.shuffle(seed=42).select(range(sample_count))
            )

    reduced_dataset = Dataset.from_dict(
        {
            k: sum([d[k] for d in reduced_dataset_list], [])
            for k in reduced_dataset_list[0].column_names
        }
    )
    return reduced_dataset


def preprocess_datasets(num_per_score, max_words, fullEvalSet=False):

    eval_num_per_score = num_per_score
    if fullEvalSet == True:
        eval_num_per_score = 10000

    reduced_dataset_train = subsample_dataset(
        dataset, split="train", num_per_score=num_per_score, max_words=max_words
    )
    reduced_dataset_eval = subsample_dataset(
        dataset, split="eval", num_per_score=eval_num_per_score, max_words=max_words
    )

    def convert_label(record):
        record["label"] = int(record["score"])
        return record

    train_dataset = reduced_dataset_train.map(convert_label)
    eval_dataset = reduced_dataset_eval.map(convert_label)

    columns_to_keep = ["text", "label"]
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in columns_to_keep]
    )
    eval_dataset = eval_dataset.remove_columns(
        [col for col in eval_dataset.column_names if col not in columns_to_keep]
    )

    train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
    eval_dataset = eval_dataset.map(lambda x: {"label": int(x["label"])})
    return train_dataset, eval_dataset


def compute_qwk(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    return {"qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic")}


def train_and_evaluate():
    global best_qwk_score, best_model_dir, best_hyperparameters
    with wandb.init(project=wandb_project, entity=wandb_entity) as run:
        config = wandb.config

        # Set the seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Preprocess datasets with current hyperparameters
        train_dataset, eval_dataset = preprocess_datasets(
            config.num_per_score, max_words, fullEvalSet=False
        )
        _, full_eval_dataset = preprocess_datasets(
            config.num_per_score, max_words, fullEvalSet=True
        )

        # Load model with hyperparameters
        # model_id = "allenai/longformer-base-4096"
        model_id = 'Leo1212/longformer-base-4096-sentence-transformers-all-nli-stsb-quora-nq'
        model = SetFitModel.from_pretrained(
            model_id, head_params={"max_iter": config.max_iter, "solver": config.solver}
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Adjust batch size if necessary
        if config.num_per_score >= 130 and config.batch_size > 2:
            config.batch_size = 2

        # Select the loss class based on the configuration
        loss_class_mapping = {
            "CosineSimilarityLoss": CosineSimilarityLoss,
            "ContrastiveLoss": ContrastiveLoss,
            "BatchAllTripletLoss": BatchAllTripletLoss,
            "BatchHardTripletLoss": BatchHardTripletLoss,
        }
        selected_loss_class = loss_class_mapping[config.loss_class]

        args = TrainingArguments(
            report_to="wandb",
            use_amp=config.use_amp,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            loss=selected_loss_class,
            body_learning_rate=config.body_learning_rate,
            num_iterations=config.num_iterations,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            warmup_proportion=config.warmup_proportion,
            load_best_model_at_end=True,
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric=compute_qwk,
            column_mapping={"text": "text", "label": "label"},
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        metrics = trainer.evaluate(full_eval_dataset)

        qwk_score = metrics.get("qwk", -1)

        wandb.log({"eval_qwk": qwk_score})
        wandb.log({"hyperparameters": dict(config)})

        print(f"Eval QWK Score: {qwk_score:.4f}")

        # Check if this is the best model so far
        if qwk_score > best_qwk_score:
            best_qwk_score = qwk_score
            best_hyperparameters = dict(config)
            best_model_dir = os.path.join(wandb.run.dir, "best_model")
            model.save_pretrained(best_model_dir)
            print(f"New best QWK: {qwk_score:.4f}. Model saved to {best_model_dir}")


# Updated sweep configuration with new hyperparameters
sweep_config = {
    "method": "random",
    "metric": {"name": "eval_qwk", "goal": "maximize"},
    "parameters": {
        "body_learning_rate": {"min": 1e-6, "max": 1e-3, "distribution": "log_uniform"},
        "num_epochs": {"min": 1, "max": 3, "distribution": "int_uniform"},
        "batch_size": {"values": [2]},
        "seed": {"min": 1, "max": 40, "distribution": "int_uniform"},
        "max_iter": {"min": 50, "max": 300, "distribution": "int_uniform"},
        "solver": {"values": ["newton-cg", "lbfgs", "liblinear"]},
        "num_iterations": {"values": [10, 20, 30, 40, 50]},
        "use_amp": {"values": [True, False]},
        "num_per_score": {"values": [80, 100, 120, 130]},
        "loss_class": {
            "values": [
                "CosineSimilarityLoss",
                "ContrastiveLoss",
                "BatchAllTripletLoss",
                "BatchHardTripletLoss",
            ]
        },
        "warmup_proportion": {"values": [0.0, 0.1, 0.2]},
    },
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project)
wandb.agent(sweep_id, train_and_evaluate)

# Push the best model to Hugging Face after the sweep
if best_model_dir:
    print(f"Pushing the best model with QWK {best_qwk_score:.4f} to Hugging Face")
    model = SetFitModel.from_pretrained(best_model_dir)
    model.push_to_hub(f"{huggingface_username}/{our_model_name}", private=True)
    print(best_hyperparameters)

    examples = []
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 1)["full_text"][0]
        .replace("\n", "")
    )
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 2)["full_text"][0]
        .replace("\n", "")
    )
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 3)["full_text"][0]
        .replace("\n", "")
    )
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 4)["full_text"][0]
        .replace("\n", "")
    )
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 5)["full_text"][0]
        .replace("\n", "")
    )
    examples.append(
        dataset["train"]
        .filter(lambda x: x["score"] == 6)["full_text"][0]
        .replace("\n", "")
    )

    # Define your variables
    model_name = f"{huggingface_username}/{our_model_name}"

    # Check if the directory already exists, if so, delete to allow for updates
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    repo = Repository(local_dir=model_name, clone_from=model_name)

    # Create or update the model card with desired information
    model_card_content = f"""
    ---  # (Rest of the model card content remains the same)
    """

    # Write content to the model card
    with open(f"{repo.local_dir}/README.md", "w") as file:
        file.write(model_card_content)

    # Push the changes to the hub
    repo.push_to_hub()
