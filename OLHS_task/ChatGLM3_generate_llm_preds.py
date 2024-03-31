import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import time

tokenizer = AutoTokenizer.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True)
model = AutoModel.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True).half().cuda()
model = model.eval()

PATH_OLHS_DATA = '../../data/labelled_data.npy'

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
    response, _ = model.chat(tokenizer, query, temperature=1.0)
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

with open('OLHS_data_ChatGLM3.json', 'w') as file:
    print(pred_list)
    json.dump(pred_list, file)
    print("have write pred_list to data.json")

with open('OLHS_data_ChatGLM3.json', 'r') as file:
    data_read = json.load(file)
