import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random

# 设置随机种子以确保结果的可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用设置随机种子函数
set_seed(42)

class FluencyScorer(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(FluencyScorer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # 获取BERT模型的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 通常使用池化输出，也可以选择使用其他层如最后一层隐藏状态的输出
        pooled_output = outputs[1]
        scores = self.linear(pooled_output)
        return scores

# 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = FluencyScorer()

# 模型评分输出
model.eval()
with torch.no_grad():
    sentences = [
        "这是一个测试句子。",
        "各个国家有各个国家的国歌",
        "各个国家有各个国家德国个"
    ]

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        score = model(inputs.input_ids, inputs.attention_mask)
        print(f'"{sentence}" Fluency Score: {score.item()}')

