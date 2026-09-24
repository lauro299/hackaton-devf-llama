import argparse
import glob
import os

import torch
from transformers import pipeline

PROMPT = """
    Transcribe todo el texto visible en esta imagen.
    Conserva la estructura orignial usando sintaxis Markdown:
    usa #/## para titulos, - o 1. para listas, ** ** para negritas, 
    y tablas Markdown si hay tablas en la imagen.
    No agregues comentarios, explicaciones, ni texto que no este en la imagen.
    Devuelve unicamente el contenido transcrito en Markdown.
    """

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_pipeline():
    return pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )


def extract_text(pip, path_image: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path_image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    output = pip(
        messages,
        return_full_text=False,
        generate_kwargs={
            "max_new_tokens": 3072,
            "repetition_penalty": 1.15,
        },
    )
    return output[0]["generated_text"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="./resources/*.jpg", help="Patron glob de las imagenes"
    )
    parser.add_argument(
        "--output", default="resultado.md", help="Archivo Mardown de salida"
    )
    parser.add_argument(
        "--separator",
        action="store_true",
        help="Si se pasa, agrega un comentario con el nombre del archivo antes de cada bloque",
    )
    args = parser.parse_args()

    images = sorted(glob.glob(args.input))

    if not images:
        print(f"No se enctoraton imagenes con el patron: {args.input}")
        return

    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    print(f"Cargando {MODEL_ID}")

    pipe = load_pipeline()

    with open(args.output, "w", encoding="utf-8") as f:
        for i, path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] processing {path}...")
            text = extract_text(pipe, path)
            if args.separator:
                name = os.path.basename(path)
                block = f"<!-- fuente: {name} -->\n\n{text}"
            else:
                block = text
            if i > 1:
                f.write("\n\n")
            f.write(block)
            f.flush()

    print(f"\nListo. Texto combinado guardado en: {args.output}")


if __name__ == "__main__":
    main()
