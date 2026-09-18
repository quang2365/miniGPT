from openai import AsyncOpenAI
import asyncio
from datasets import load_dataset
from pydantic import BaseModel, Field
import pathlib
import json
class Datasetformat(BaseModel):
    prompt: str = Field(description='translated prompt')
    chosen: str = Field(description='translated chosen')
    rejected: str = Field(description='translated rejected')

model = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='ollama', timeout=None)

DATASET_PATH = pathlib.Path(__file__).resolve().parent / 'data' / 'data' / 'train-00000-of-00001-9dffc9d46d32c335.parquet'
SAVE_PATH = pathlib.Path(__file__).resolve().parent / 'data' / 'data' / 'dataset.json'
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

sem = asyncio.Semaphore(4)
file_lock = asyncio.Lock()

ds = load_dataset('parquet', data_files=str(DATASET_PATH))

instructions = ds['train']['instruction']
chosen = ds['train']['chosen_response']
rejected = ds['train']['rejected_response']
num_sample = min(3000,len(instructions))

async def save_result(idx, result):
    async with file_lock:
        data = []
        if SAVE_PATH.exists():
            try:
                with open(SAVE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []

        if idx >= len(data):
            data.extend([None] * (idx - len(data) + 1))
        data[idx] = result

        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(idx):
    message = [{'role':'user','content':f"""translate into vietnamese:
        prompt:{instructions[idx]}
        chosen:{chosen[idx]}
        rejected:{rejected[idx]}"""}]
    return message
async def process_single_row(idx):
    async with sem:
        message = build_prompt(idx)
        translate_response = await model.beta.chat.completions.parse(
            model='qwen2.5:7b',
            messages=message,
            temperature=0.1,
            max_tokens=30000,
            response_format=Datasetformat,
            extra_body={
                'options': {
                    'think': False
                }
            }
        )
        result = translate_response.choices[0].message.parsed.model_dump()
        print(f'done sample {idx}')

    await save_result(idx, result)
    return result


async def main():
    tasks = [process_single_row(i) for i in range(555,num_sample)]
    return await asyncio.gather(*tasks)


new_dataset = asyncio.run(main())
