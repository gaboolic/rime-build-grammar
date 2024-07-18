import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 示例数据集定义
class ExampleDataset(Dataset):
    def __init__(self, tokenizer, sentences, scores):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.scores = scores

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.sentences[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        labels = torch.tensor(self.scores[idx], dtype=torch.float)
        return {'inputs': inputs, 'labels': labels}

# 假设你有以下数据集
sentences = ["这是一个测试句子。", "各个国家有各个国家的国歌", "各个国家有各个国家德国个","请务必在抽血时间截止前到达门店","请留意门店短信提示或查看订单详情"]
scores = [1, 1, 0.2, 1,1]  # 用实际流畅度评分数据替换这些分数

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = ExampleDataset(tokenizer, sentences, scores)

# 自定义 collate_fn 以适配 DataCollatorWithPadding
def custom_collate_fn(batch):
    batch_inputs = [item['inputs'] for item in batch]
    batch_labels = torch.stack([item['labels'] for item in batch])
    inputs = data_collator(batch_inputs)
    return {'inputs': inputs, 'labels': batch_labels}

data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

# 定义模型
class FluencyScorer(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(FluencyScorer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        scores = self.linear(pooled_output)
        return scores

model = FluencyScorer()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
epochs = 3
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        inputs = batch['inputs']
        labels = batch['labels'].to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        optimizer.zero_grad()
        outputs = model(**inputs).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


test_sentences = ["这是一个测试句子", "各个国家有各个国家的国歌", "各个国家有各个国家德国个","禁止早恋","进制造连","可以加辅助码","可以贾府竹马","可以家父猪吗",
                  "充满希望的跋涉比到达目的地更能给人乐趣",
                  "充满希望的跋涉比到达目的地更能给人了去",
                  "充满希望的吧涉笔到达目的地更能给人了去"
                  ]

# 进行测试
model.eval()

with torch.no_grad():
    for test_sentence in test_sentences:
        #test_sentence = "这是一个测试句子。"
        inputs = tokenizer(test_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        score = model(inputs['input_ids'], inputs.get('attention_mask'))
        print(f"\"{test_sentence}\" Fluency Score: {score.item()}")

# 保存模型状态字典
torch.save(model.state_dict(), 'bert/fluency_scorer_model.pth')
