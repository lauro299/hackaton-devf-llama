"""Fine-tuning contrastivo de paraphrase-multilingual-MiniLM-L12-v2 sobre el
dataset de (pregunta, chunk-correcto) generado por build_finetune_dataset.py.

Usa MultipleNegativesRankingLoss: dentro de cada batch, el texto positivo de
las demas preguntas actua como negativo implicito -- no hace falta minar
negativos duros a mano para un dataset de este tamano.

Este script NO reemplaza el modelo actual en rag_cli.py/build_index.py --
el resultado se guarda en un directorio aparte. Para usarlo, apuntar
`--model models/embedder-book-finetuned` al reconstruir book.db con
build_index.py (y pasar `--embedder` con la misma ruta a rag_cli.py).

Uso:
    python app/build_finetune_dataset.py   # genera data/embedder_finetune_{train,eval}.jsonl
    python app/finetune_embedder.py
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TRAIN_PATH = Path("data/embedder_finetune_train.jsonl")
EVAL_PATH = Path("data/embedder_finetune_eval.jsonl")
DEFAULT_OUTPUT_DIR = "models/embedder-book-finetuned"


def load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not TRAIN_PATH.exists() or not EVAL_PATH.exists():
        raise SystemExit(
            f"No encuentro {TRAIN_PATH} / {EVAL_PATH} -- corre primero "
            "app/build_finetune_dataset.py."
        )

    train_dataset = load_jsonl(TRAIN_PATH)
    eval_dataset = load_jsonl(EVAL_PATH)
    print(f"Train: {len(train_dataset)} pares -- Eval: {len(eval_dataset)} pares")

    model = SentenceTransformer(BASE_MODEL)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    model.save(args.output)
    print(f"Modelo guardado en: {args.output}")


if __name__ == "__main__":
    main()
