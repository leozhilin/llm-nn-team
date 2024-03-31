from transformers import AutoTokenizer, AutoModel
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

tokenizer = AutoTokenizer.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True)
model = AutoModel.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True).half().cuda()
model = model.eval()
Data = {}
with open('./xsum.json', 'rt', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        sample = json.loads(line.strip())
        Data[idx] = sample


pred_list = []
for i in range(len(Data)):
    if i >= 0:
        text = Data[i]['dialogue']
        ground_truth = Data[i]['summary']
        start_time = time.time()
        prompt = "Please generate a one-sentence summary for the given document. "
        if (len(prompt + text) < 8000):
            ans, history = model.chat(tokenizer, prompt + text, history=[], temperature=1.0)
            end_time = time.time()
            # print(ans)
            gen = {"id": i, "input": text, "ground_truth": ground_truth, "generation": ans, "time": end_time - start_time}
            print(gen)
            dump_jsonl(gen, "ChatGLM3.json")
