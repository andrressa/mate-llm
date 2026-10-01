from datasets import load_dataset
import random
import re
from pathlib import Path


DATASET_NAME = "zwhe99/DeepMath-103K"
SPLIT = "train"
OLD_SAMPLE_FILE = r"C:\Users\andrr\Downloads\deepmath_samples\deepmath_100.txt"

OUTPUT_FILE = r"C:\Users\andrr\Downloads\deepmath_samples\deepmath_heldout_100.txt"

N = 100
SEED = 2026



def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_previous_questions(path):

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"QUESTION\s+\d+\s*\n", text)[1:]

    previous = set()

    for block in blocks:

        block = re.split(r"\n={5,}\n", block)[0]

        q = normalize_text(block)

        if q:
            previous.add(q)

    return previous


print("Carregando dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT
)

print("Total de registros:", len(dataset))
print("Colunas disponíveis:", dataset.column_names)



possible_columns = [
    "question",
    "problem",
    "prompt",
    "text"
]

question_column = None

for col in possible_columns:
    if col in dataset.column_names:
        question_column = col
        break


if question_column is None:
    raise ValueError(
        "Não foi possível identificar automaticamente "
        "a coluna dos enunciados.\n"
        f"Colunas disponíveis: {dataset.column_names}"
    )


print("Coluna utilizada:", question_column)



previous_questions = read_previous_questions(OLD_SAMPLE_FILE)

print(
    len(previous_questions)
)



candidates = []

seen = set()

for row in dataset:

    question = row.get(question_column)

    if question is None:
        continue

    question = normalize_text(question)

    if not question:
        continue


    if question in previous_questions:
        continue

    if question in seen:
        continue

    seen.add(question)

    candidates.append(question)


print(
    len(candidates)
)


if len(candidates) < N:

    raise ValueError(
        f"Foram encontradas apenas {len(candidates)} "
        f"questões inéditas, mas são necessárias {N}."
    )




random.seed(SEED)

heldout = random.sample(
    candidates,
    N
)



assert len(heldout) == 100

assert len(set(heldout)) == 100

assert len(set(heldout) & previous_questions) == 0


print("\nVERIFICAÇÃO FINAL")
print("-----------------------------")
print("Questões selecionadas:", len(heldout))
print("Duplicatas internas:", len(heldout) - len(set(heldout)))
print(
    "Questões repetidas do conjunto anterior:",
    len(set(heldout) & previous_questions)
)



with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for i, question in enumerate(
        heldout,
        start=1
    ):

        f.write(f"QUESTION {i}\n")

        f.write(question)

        f.write("\n\n")

        f.write("=" * 80)

        f.write("\n\n")


print("\nArquivo criado com sucesso:")
print(OUTPUT_FILE)