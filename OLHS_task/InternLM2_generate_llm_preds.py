import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import time
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
model = model.eval()
# (Optional) If on low resource devices, you can load model in 4-bit or 8-bit to further save GPU memory via bitsandbytes.
  # InternLM 7B in 4bit will cost nearly 8GB GPU memory.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)


PATH_OLHS_DATA = '../../data/labelled_data.npy'
# PATH_GLOVE_MODEL = '../data/glove.6B.100d.txt'

data = np.load(PATH_OLHS_DATA, allow_pickle=True)[()]
posts = data['posts']
labels = data['labels']
labels = [1 if int(l) < 2 else 0 for l in labels]

pred_list = []
for i, j in enumerate(zip(posts, labels)):
    start_time = time.time()
    query = f"""
            The following is an excerpt from a comment on Twitter that may contain aggressive or sentiment speech. 
            You need to analyze the emotional tendency of the content of this comment.
            Reply with 1 if the comment is considered aggressive or sentiment speech, and reply with 0 if it is neither.
            You only need to reply with 0 or 1, and no other responses are allowed.
            here is the sentence：{' '.join(j[0])}
            """
    response, history = model.chat(tokenizer, query, history=[], temperature=1.0)
    # print(response)
    end_time = time.time()
    try:
        if int(response) == 0:
            pred = 0
        else:
            pred = 1
    except:
        pred = 0
        pass
    tmp = {}
    tmp["Id"] = i
    tmp["Text"] = ' '.join(j[0])
    tmp["Label"] = int(j[1])
    tmp["Preds"] = pred
    tmp["Time"] = end_time - start_time
    print(i, tmp["Label"], tmp["Preds"], tmp["Time"])
    pred_list.append(tmp)

with open('OLHS_data_InternLM2.json', 'w') as file:
    print(pred_list)
    json.dump(pred_list, file)
    print("have write pred_list to data.json")

# 从文件中读取列表
with open('OLHS_data_InternLM2.json', 'r') as file:
    data_read = json.load(file)

# 打印读取的列表
for data in data_read:
    print(data["Text"])

# [
#     {
#         "Id": "样本编号，这不是必需的",
#         "Text": "数据内容",
#         "Label": "数据标签",
#         "Preds": "LLM生成的预测",
#         "Time": "LLM预测所花费的时间"
#     }
# ]
#
# [
#     {
#         "Id": "样本编号，这不是必需的",
#         "Text": "数据内容",
#         "Label": "数据标签",
#         "Preds": "LLM生成的预测",
#         "Completion_tokens": "回复所花费的token数",
#         "Prompt_tokens": "prompt所花费的token数"
#     }
# ]