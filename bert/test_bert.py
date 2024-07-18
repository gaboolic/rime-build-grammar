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
# 初始化模型
model.load_state_dict(torch.load('temp/fluency_scorer_model.pth'))
model.to(device)



test_sentences = ["这是一个测试句子", "各个国家有各个国家的国歌", "各个国家有各个国家德国个","禁止早恋","进制造连","可以加辅助码","可以贾府竹马","可以家父猪吗",
                  "充满希望的跋涉比到达目的地更能给人乐趣",
                  "充满希望的跋涉比到达目的地更能给人了去",
                  "充满希望的吧涉笔到达目的地更能给人了去"
                  ]

# 进行测试
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
with torch.no_grad():
    for test_sentence in test_sentences:
        #test_sentence = "这是一个测试句子。"
        inputs = tokenizer(test_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        score = model(inputs['input_ids'], inputs.get('attention_mask'))
        print(f"\"{test_sentence}\" Fluency Score: {score.item()}")

