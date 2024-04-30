import json
import random
import numpy as np
import pandas as pd
import statistics as statistics
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time


def random_zero_or_one(p):
    random_number = random.random()
    if random_number < p:
        return 0
    else:
        return 1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class OLHS_Dataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.targets = data["Label"].values
        self.llm_preds = data["Pred"].values
        self.tokens = data["Token"].values
        self.posts_features = []
        for features in data["Feature"].values:
            features = torch.from_numpy(features).float()
            features = features.to(device)
            self.posts_features.append(features)
        print("length:", len(self.posts_features))

    def __getitem__(self, index: int):
        target, llm_pred = self.targets[index], self.llm_preds[index]
        post_features = self.posts_features[index]
        token = self.tokens[index]
        return post_features, target, llm_pred, token

    def __len__(self) -> int:
        return len(self.posts_features)


class OLHS_3_Split_Dataloader:
    def __init__(self, train_batch_size=128, test_batch_size=128, seed=42):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed

        dataset = []
        with open(OLHS_WITH_PREDS_PATH, 'r') as file:
            for idx, line in enumerate(file):
                sample = json.loads(line.strip())
                dataset.append(sample)

        print("Loading Glove Model")
        f = open(PATH_GLOVE_MODEL, 'r', errors='ignore')
        GLOVE_MODEL = {}
        for line in f:
            split_lines = line.split()
            word = split_lines[0]
            word_embedding = np.array([float(value) for value in split_lines[1:]])
            GLOVE_MODEL[word] = word_embedding
        vocab = GLOVE_MODEL.keys()
        targets = []
        labels = []
        preds = []
        tokens = []
        # 打印读取的列表
        for data in dataset:
            features = []
            for i in data["Text"].split():
                if i in vocab:
                    features.append(GLOVE_MODEL[i])
            if len(features) == 0:
                continue
            features = np.mean(features, axis=0)
            data["Feature"] = features

            targets.append(features)
            labels.append(data["Label"])
            tokens.append(data["Completion_tokens"] + data["Prompt_tokens"])
            if (random_zero_or_one(1) == 0):
                preds.append(data["Preds"])
            else:
                preds.append(random_zero_or_one(0.5))

        data_len = len(dataset)

        train_labels = labels[:int(data_len * 0.8)]
        train_features = targets[:int(data_len * 0.8)]
        train_preds = preds[:int(data_len * 0.8)]
        train_tokens = tokens[:int(data_len * 0.8)]

        test_labels = labels[int(data_len * 0.8):]
        test_features = targets[int(data_len * 0.8):]
        test_preds = preds[int(data_len * 0.8):]
        test_tokens = tokens[int(data_len * 0.8):]

        train_df = pd.DataFrame(
            {"Label": train_labels, "Feature": train_features, "Pred": train_preds, "Token": train_tokens})
        test_df = pd.DataFrame(
            {"Label": test_labels, "Feature": test_features, "Pred": test_preds, "Token": test_tokens})

        self.trainset = OLHS_Dataset(train_df)
        self.testset = OLHS_Dataset(test_df)

    def get_data_loader(self):
        train_loader = self._get_data_loader(dataset=self.trainset, batch_size=self.train_batch_size, drop_last=True)
        test_loader = self._get_data_loader(dataset=self.testset, batch_size=1, drop_last=False)
        return train_loader, test_loader

    def _get_data_loader(self, dataset, batch_size, drop_last, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                                           drop_last=drop_last)


class Network(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(100, NUM_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_UNITS, output_size)
        )

    def forward(self, features):
        output = self.classifier(features)
        output = nn.Softmax(dim=1)(output)
        return output


def CET_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets):
    # Input:
    #   epoch: int = current epoch (not used)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: softmax outputs as weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(
        classifier_output.device)  # set up zero-initialized tensor to store team predictions
    team_probs = team_probs + allocation_system_output[:, 0].reshape(-1,
                                                                     1) * classifier_output  # add the weighted classifier prediction to the team prediction
    for idx in range(NUM_EXPERTS):  # continue with human experts
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[expert_preds[idx].astype(int)]).to(
            classifier_output.device)
        team_probs = team_probs + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    system_loss = nn.NLLLoss()(log_output, targets)

    return system_loss


def generate_allocation_system_data(epoch, classifier_output, input, targets, preds, over_under_sampling='without'):
    classifier_preds_list = classifier_output.argmax(1).cpu().tolist()
    batch_features_list = input.cpu().tolist()
    true_labels = targets.cpu().tolist()
    if over_under_sampling == 'without':
        h = []
        allocation_inputs = []
        for i, (feature, classifier_pred, target, llm_pred) in enumerate(zip(batch_features_list, classifier_preds_list,
                                                                             true_labels, preds)):
            if i == 0:
                allocation_inputs.append(feature)
                h.append(0)
            if classifier_pred != llm_pred:
                if classifier_pred == target:
                    allocation_inputs.append(feature)
                    h.append(0)
                elif llm_pred == target:
                    allocation_inputs.append(feature)
                    h.append(1)
        allocation_inputs = torch.tensor(allocation_inputs).to(device)
        h = torch.tensor(h).to(device)
        return allocation_inputs, h
    else:
        h_llm = []
        h_classifier = []
        allocation_inputs_llm = []
        allocation_inputs_classifier = []
        for i, (feature, classifier_pred, target, llm_pred) in enumerate(zip(batch_features_list, classifier_preds_list,
                                                                             true_labels, preds)):
            if classifier_pred != llm_pred:
                if llm_pred == target:
                    allocation_inputs_llm.append(feature)
                    h_llm.append(1)
                elif classifier_pred == target:
                    allocation_inputs_classifier.append(feature)
                    h_classifier.append(0)

        len_llm = len(h_llm)
        len_classifier = len(h_classifier)
        if len_llm < len_classifier:
            h_llm = h_llm + h_llm[:len_classifier - len_llm]
            allocation_inputs_llm = allocation_inputs_llm + allocation_inputs_llm[:len_classifier - len_llm]
        else:
            h_classifier = h_classifier + h_classifier[:len_llm - len_classifier]
            allocation_inputs_classifier = allocation_inputs_classifier + allocation_inputs_classifier[
                                                                          :len_llm - len_classifier]

        allocation_inputs = allocation_inputs_llm + allocation_inputs_classifier
        h = h_llm + h_classifier

        if len(allocation_inputs):
            combined = list(zip(allocation_inputs, h))
            random.shuffle(combined)
            allocation_inputs, h = zip(*combined)
            allocation_inputs = list(allocation_inputs)
            h = list(h)

        allocation_inputs = torch.tensor(allocation_inputs).to(device)
        h = torch.tensor(h).to(device)
        return allocation_inputs, h

def get_accuracy(preds, targets):
    if len(targets) > 0:
        acc = accuracy_score(targets, preds)
    else:
        acc = 0
    return acc

def train_classifier(epoch, classifier, train_loader, classifier_optimizer, classifier_scheduler, classifier_loss_fn):
    classifier.train()

    allocation_input_list = []
    h_list = []
    for i, (batch_features, batch_targets, batch_preds, _) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)
        batch_outputs_classifier = classifier(batch_features)

        classifier_loss = classifier_loss_fn(batch_outputs_classifier, batch_targets)

        classifier_optimizer.zero_grad()
        classifier_loss.backward()
        classifier_optimizer.step()
        if USE_LR_SCHEDULER:
            classifier_scheduler.step()

        allocation_inputs, h = generate_allocation_system_data(epoch, batch_outputs_classifier, batch_features,
                                                               batch_targets, batch_preds)
        allocation_input_list.append(allocation_inputs)
        h_list.append(h)
    return allocation_input_list, h_list

def train_allocation(allocation_input_list, h_list, allocation_system, allocation_system_optimizer,
                     allocation_system_loss_fn, allocation_system_scheduler):
    allocation_system.train()
    for _ in range(ALLOCATION_EPOCHS):
        for i, (allocation_inputs, h) in enumerate(zip(allocation_input_list, h_list)):
            batch_outputs_allocation_system = allocation_system(allocation_inputs)
            allocation_system_loss = allocation_system_loss_fn(batch_outputs_allocation_system, h)
            allocation_system_optimizer.zero_grad()
            allocation_system_loss.backward()
            allocation_system_optimizer.step()
            if USE_LR_SCHEDULER:
                allocation_system_scheduler.step()

def evaluate_allocation(epoch, classifier, allocation_system, test_loader):
    allocation_system.eval()
    classifier.eval()
    system_decisions = []
    targets = []
    sum0 = 0
    sum1 = 0
    llm_token = torch.tensor([0]).float()
    with torch.no_grad():
        for i, (batch_features, batch_targets, batch_preds, token) in enumerate(test_loader):
            batch_features = batch_features.to(device)

            batch_allocation_system_outputs = allocation_system(batch_features)
            batch_system_decisions = np.argmax(batch_allocation_system_outputs.to("cpu"), 1)
            targets.append(batch_targets)
            if batch_system_decisions == 1:
                sum1 += 1
                system_decisions.append(batch_preds)
                llm_token += token
            else:
                sum0 += 1
                batch_classifier_outputs = classifier(batch_features)
                classifier_preds = np.argmax(batch_classifier_outputs.to("cpu"), 1)
                system_decisions.append(classifier_preds)
        system_decisions = torch.cat(system_decisions, dim=0)
        targets = torch.cat(targets, dim=0)
    system_accuracy = get_accuracy(system_decisions, targets)
    token_consumption = llm_token.item()
    print(f"choose llm: {sum1}, choose classifier: {sum0}")
    return system_accuracy, token_consumption, sum0, sum1

def evaluate_classifier(epoch, classifier, test_loader):
    classifier.eval()
    classifier_decisions = []
    targets = []
    with torch.no_grad():
        for batch_features, batch_targets, batch_preds, batch_time_cost in test_loader:
            batch_classifier_outputs = classifier(batch_features)
            batch_classifier_decisions = np.argmax(batch_classifier_outputs.to("cpu"), 1)
            targets.append(batch_targets)
            classifier_decisions.append(batch_classifier_decisions)
        classifier_decisions = torch.cat(classifier_decisions, dim=0)
        targets = torch.cat(targets, dim=0)
        classifier_accuracy = get_accuracy(classifier_decisions, targets)
        end_time = time.time()
    return classifier_accuracy, 0

def evaluate_llm(test_loader):
    llm_decisions = []
    targets = []
    llm_token = torch.tensor([0]).float()
    with torch.no_grad():
        for batch_features, batch_targets, batch_preds, batch_token in test_loader:
            llm_token += batch_token
            targets.append(batch_targets)
            llm_decisions.append(batch_preds)
        targets = torch.cat(targets, dim=0)
        llm_decisions = torch.cat(llm_decisions, dim=0)
        llm_accuracy = get_accuracy(llm_decisions, targets)
    return llm_accuracy, llm_token.item()


def my_approach(train_loader, test_loader):
    classifier = Network(output_size=NUM_CLASSES).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)
    classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, EPOCHS * len(train_loader))
    classifier_loss_fn = nn.CrossEntropyLoss()

    allocation_system = Network(output_size=NUM_EXPERTS + 1).to(device)
    allocation_system_optimizer = torch.optim.Adam(allocation_system.parameters(), lr=LR, betas=(0.9, 0.999),
                                                   weight_decay=0)
    allocation_system_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(allocation_system_optimizer,
                                                                             EPOCHS * len(train_loader))
    allocation_system_loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_token_consumption = 1e9

    for epoch in range(1, EPOCHS + 1):
        print("current epoch:", epoch)
        allocation_input_list, h_list = train_classifier(epoch, classifier, train_loader, classifier_optimizer, classifier_scheduler, classifier_loss_fn)
        train_allocation(allocation_input_list, h_list, allocation_system, allocation_system_optimizer,
                         allocation_system_loss_fn, allocation_system_scheduler)
        system_accuracy, system_token_consumption, sum0, sum1 = evaluate_allocation(epoch, classifier, allocation_system, test_loader)
        print("system_accuracy:", system_accuracy, "system_token_consumption:", system_token_consumption)
        classifier_accuracy, classifier_token_consumption = evaluate_classifier(epoch, classifier, test_loader)
        print("classifier_accuracy:", classifier_accuracy, "classifier_token_consumption:", classifier_token_consumption)
        llm_accuracy, llm_token_consumption = evaluate_llm(test_loader)
        print("llm_accuracy:", llm_accuracy, "llm_token_consumption:", llm_token_consumption)

        if system_accuracy > best_accuracy:
            best_accuracy = system_accuracy
            best_accuracy_result = [system_accuracy, system_token_consumption, sum0, sum1]
        if system_token_consumption < best_token_consumption:
            best_token_consumption = system_token_consumption
            best_token_consumption_result = [system_accuracy, system_token_consumption, sum0, sum1]
    return best_accuracy_result, best_token_consumption_result


def C_E_Team_train_one_epoch(epoch, classifier, allocation_system, train_loader, optimizer, scheduler, loss_fn):
    classifier.train()
    allocation_system.train()

    for i, (batch_features, batch_targets, batch_preds, _) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)
        llm_batch_preds = batch_preds.numpy().reshape(1, -1)

        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier,
                             allocation_system_output=batch_outputs_allocation_system, expert_preds=llm_batch_preds,
                             targets=batch_targets)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def C_E_Team_evaluate_one_epoch(epoch, classifier, allocation_system, data_loader, loss_fn):
    classifier.eval()
    allocation_system.eval()

    classifier_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    llm_preds = torch.tensor([])
    targets = torch.tensor([]).long().to(device)

    with torch.no_grad():
        for i, (batch_features, batch_targets, batch_preds, _) in enumerate(data_loader):
            batch_targets = batch_targets.to(device)
            targets = torch.cat((targets, batch_targets))

            batch_classifier_outputs = classifier(batch_features)
            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))

            batch_allocation_system_outputs = allocation_system(batch_features)
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))

            llm_preds = torch.cat((llm_preds, batch_preds))

    llm_preds = llm_preds.numpy().reshape(1, -1)

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)

    preds = np.vstack((classifier_preds, llm_preds)).T

    system_preds = preds[range(len(preds)), allocation_system_decisions.astype(int)]
    system_accuracy = get_accuracy(system_preds, targets)
    llm_accuracy = get_accuracy(np.squeeze(llm_preds), targets)
    nn_accuracy = get_accuracy(classifier_preds, targets)

    return system_accuracy, llm_accuracy, nn_accuracy


def C_E_Team(train_loader, test_loader):
    print("C_E_Team:")
    classifier = Network(output_size=NUM_CLASSES).to(device)
    allocation_system = Network(output_size=NUM_EXPERTS + 1).to(device)

    parameters = list(allocation_system.parameters())
    parameters += list(classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))
    loss_fn = CET_loss

    best_accuracy = 0
    best_token_consumption = 1e9
    for epoch in range(1, EPOCHS + 1):
        print("current epoch:", epoch)
        C_E_Team_train_one_epoch(epoch, classifier, allocation_system, train_loader, optimizer, scheduler, loss_fn)
        system_accuracy, system_token_consumption, sum0, sum1 = evaluate_allocation(epoch, classifier, allocation_system, test_loader)
        print("system_accuracy:", system_accuracy, "system_token_consumption", system_token_consumption)
        classifier_accuracy, classifier_token_consumption = evaluate_classifier(epoch, classifier, test_loader)
        print("classifier_accuracy:", classifier_accuracy, "classifier_token_consumption:", classifier_token_consumption)
        llm_accuracy, llm_token_consumption = evaluate_llm(test_loader)
        print("llm_accuracy:", llm_accuracy, "llm_token_consumption:", llm_token_consumption)

        if system_accuracy > best_accuracy:
            best_accuracy = system_accuracy
            best_accuracy_result = [system_accuracy, system_token_consumption, sum0, sum1]
        if system_token_consumption < best_token_consumption:
            best_token_consumption = system_token_consumption
            best_token_consumption_result = [system_accuracy, system_token_consumption, sum0, sum1]
    return best_accuracy_result, best_token_consumption_result

def classifier_baseline(train_loader, test_loader):
    classifier = Network(output_size=NUM_CLASSES).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)
    classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, EPOCHS * len(train_loader))
    classifier_loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        for i, (batch_features, batch_targets, batch_preds, _) in enumerate(train_loader):
            batch_targets = batch_targets.to(device)
            batch_outputs_classifier = classifier(batch_features)

            classifier_loss = classifier_loss_fn(batch_outputs_classifier, batch_targets)

            classifier_optimizer.zero_grad()
            classifier_loss.backward()
            classifier_optimizer.step()
            if USE_LR_SCHEDULER:
                classifier_scheduler.step()
        classifier_accuracy, _ = evaluate_classifier(epoch, classifier, test_loader)
        best_accuracy = max(best_accuracy, classifier_accuracy)
    return best_accuracy, 0

if __name__ == '__main__':

    PATH_GLOVE_MODEL = '../../data/glove.6B.100d.txt'  #你需要将此路径替换为实际GloVe文件所在的路径
    OLHS_WITH_PREDS_PATH = '../../OLHS_task/OLHS_data_Online_Qwen.json' #你需要将此路径替换为数据文件所在的路径
    SEED = 67
    EXPERIMENT_EPOCH = 3
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    NUM_CLASSES = 2
    NUM_EXPERTS = 1
    DROPOUT = 0.00
    NUM_HIDDEN_UNITS = 50
    LR = 5e-3
    EPOCHS = 2
    ALLOCATION_EPOCHS = 5
    USE_LR_SCHEDULER = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ohs_dl = OLHS_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)
    train_loader, test_loader = ohs_dl.get_data_loader()

    best_accuracy_result_list = []
    best_token_consumption_result_list = []
    for i in range(EXPERIMENT_EPOCH):
        setup_seed(SEED + i)
        best_accuracy_result, best_token_consumption_result = my_approach(train_loader, test_loader)
        best_accuracy_result_list.append(best_accuracy_result)
        best_token_consumption_result_list.append(best_token_consumption_result)

    stdev_best_accuracy_result = [statistics.stdev(data) for data in zip(*best_accuracy_result_list)]
    avg_best_accuracy_result = [sum(data) / EXPERIMENT_EPOCH for data in zip(*best_accuracy_result_list)]
    stdev_best_token_consumption_result = [statistics.stdev(data) for data in zip(*best_token_consumption_result_list)]
    avg_best_token_consumption_result = [sum(data) / EXPERIMENT_EPOCH for data in zip(*best_token_consumption_result_list)]


    CET_best_accuracy_result_list = []
    CET_best_token_consumption_result_list = []
    for i in range(EXPERIMENT_EPOCH):
        setup_seed(SEED + i)
        CET_best_accuracy_result, CET_best_time_cost_result = C_E_Team(train_loader, test_loader)
        CET_best_accuracy_result_list.append(CET_best_accuracy_result)
        CET_best_token_consumption_result_list.append(CET_best_time_cost_result)

    CET_stdev_best_accuracy_result = [statistics.stdev(data) for data in zip(*CET_best_accuracy_result_list)]
    CET_avg_best_accuracy_result = [sum(data) / EXPERIMENT_EPOCH for data in zip(*CET_best_accuracy_result_list)]
    CET_stdev_best_token_consumption_result = [statistics.stdev(data) for data in zip(*CET_best_token_consumption_result_list)]
    CET_avg_best_token_consumption_result = [sum(data) / EXPERIMENT_EPOCH for data in zip(*CET_best_token_consumption_result_list)]


    classifier_accuracy_list = []
    classifier_token_consumption_result_list = []
    for i in range(EXPERIMENT_EPOCH):
        setup_seed(SEED + i)
        accuracy, time_cost = classifier_baseline(train_loader, test_loader)
        classifier_accuracy_list.append(accuracy)
        classifier_token_consumption_result_list.append(time_cost)

    classifier_stdev_best_accuracy_result = statistics.stdev(classifier_accuracy_list)
    classifier_avg_best_accuracy_result = sum(classifier_accuracy_list) / EXPERIMENT_EPOCH
    classifier_stdev_best_token_consumption_result = statistics.stdev(classifier_token_consumption_result_list)
    classifier_avg_best_token_consumption_result = sum(classifier_token_consumption_result_list) / EXPERIMENT_EPOCH

    llm_accuracy, llm_token_consumption = evaluate_llm(test_loader)


    print("-------------------------------------------baseline--------------------------------------------")
    print("llm_accuracy:", llm_accuracy, "llm token consumption:", llm_token_consumption)
    print(f"nn accuracy : {classifier_avg_best_accuracy_result:>0.2f}({classifier_stdev_best_accuracy_result:>0.2f}), nn token consumption : {classifier_avg_best_token_consumption_result:>0.2f}({classifier_stdev_best_token_consumption_result:>0.2f})\n")

    print("------------------------------------------my approach------------------------------------------")
    print(f"best accuracy result : \n"
          f"system accuracy : {avg_best_accuracy_result[0]:>0.2f}({stdev_best_accuracy_result[0]:>0.2f}), system token consumption : {avg_best_accuracy_result[1]:>0.2f}({stdev_best_accuracy_result[1]:>0.2f}), system choose classifier/llm : {avg_best_accuracy_result[2]:>0.0f} / {avg_best_accuracy_result[3]:>0.0f}\n")

    print(f"best token consumption result : \n"
          f"system accuracy : {avg_best_token_consumption_result[0]:>0.2f}({stdev_best_token_consumption_result[0]:>0.2f}), system token consumption : {avg_best_token_consumption_result[1]:>0.2f}({stdev_best_token_consumption_result[1]:>0.2f}), system choose classifier/llm : {avg_best_token_consumption_result[2]:>0.0f} / {avg_best_token_consumption_result[3]:>0.0f}\n")

    print("-------------------------------------------C_E_Team--------------------------------------------")
    print(f"best accuracy result : \n"
          f"system accuracy : {CET_avg_best_accuracy_result[0]:>0.2f}({CET_stdev_best_accuracy_result[0]:>0.2f}), system token consumption : {CET_avg_best_accuracy_result[1]:>0.2f}({CET_stdev_best_accuracy_result[1]:>0.2f}), system choose classifier/llm : {CET_avg_best_accuracy_result[2]:>0.0f} / {CET_avg_best_accuracy_result[3]:>0.0f}\n")
    print(f"best token consumption result : \n"
          f"system accuracy : {CET_avg_best_token_consumption_result[0]:>0.2f}({CET_stdev_best_token_consumption_result[0]:>0.2f}), system token consumption : {CET_avg_best_token_consumption_result[1]:>0.2f}({CET_stdev_best_token_consumption_result[1]:>0.2f}), system choose classifier/llm : {CET_avg_best_token_consumption_result[2]:>0.0f} / {CET_avg_best_token_consumption_result[3]:>0.0f}\n")