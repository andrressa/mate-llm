from datasets import load_dataset
import json
import os

'''
The Python routine loads the 'DeepMath-103K' dataset from Hugging Face using the 'train' split in streaming mode. 
Only the 'question' field is extracted. 
The selected questions are organized into four nested subsets containing 20, 30, 40, and 50 problems. 
Each subset is then saved in both '.json' and '.txt' formats for use in the annotation experiments.
'''

DATASET_NAME = "zwhe99/DeepMath-103K"


SIZES = [100]
# SEED = 42
OUTPUT_DIR = "deepmath_samples_hold"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Load DeepMath-103K...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

dataset = dataset.shuffle(
    seed=SEED,
    buffer_size=10000
)



max_size = max(SIZES)

questions = []

for example in dataset:

    question = example["question"]

    # evita questões vazias
    if question is not None and question.strip():
        questions.append(question.strip())

    if len(questions) >= max_size:
        break


print(f"{len(questions)} questões selecionadas.")



for size in SIZES:

    sample = questions[:size]


    json_path = os.path.join(
        OUTPUT_DIR,
        f"deepmath_{size}.json"
    )

    data = [
        {
            "id": i + 1,
            "question": question
        }
        for i, question in enumerate(sample)
    ]

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )




    txt_path = os.path.join(
        OUTPUT_DIR,
        f"deepmath_{size}.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        for i, question in enumerate(sample, start=1):

            f.write(f"QUESTION {i}\n")
            f.write(question)
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\n\n")


    print(
        f"set with {size} questions:"
        f"\n  {json_path}"
        f"\n  {txt_path}"
    )


print("\n Finish.")