from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

summarizer = pipeline("summarization", model="./falconsai_text_summarization", device=0)
import json
import time
def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    # mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

Data = {}
with open('./ChatGLM3.json', 'rt', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        sample = json.loads(line.strip())
        Data[idx] = sample


for i in range(len(Data)):
    # print(Data[i])
    text = Data[i]['input']
    start_time = time.time()
    ans = summarizer(text, max_length=388, min_length=30, do_sample=False)
    end_time = time.time()

    Data[i]['nn_generation'] = ans[0]['summary_text']
    Data[i]['nn_time'] = end_time - start_time
    print(i, Data[i]['nn_generation'], Data[i]['nn_time'])
    dump_jsonl(Data[i], "NN.json")
