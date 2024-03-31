# export MODELSCOPE_CACHE='./7b_models'
# export MODELSCOPE_MODULES_CACHE='./7b_models'
# #模型下载
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-7B-Chat",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-7B-Chat")




import numpy as np
import json
import time


PATH_OLHS_DATA = './data/labelled_data.npy'
PATH_GLOVE_MODEL = './data/glove.6B.100d.txt'

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
    messages = [
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

with open('OLHS_data_Qwen.json', 'w') as file:
    print(pred_list)
    json.dump(pred_list, file)
    print("have write pred_list to data.json")

# 从文件中读取列表
with open('OLHS_data_Qwen.json', 'r') as file:
    data_read = json.load(file)

# 打印读取的列表
for data in data_read:
    print(data["Text"])

