import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel

class FluencyScorer(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(FluencyScorer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        scores = self.linear(pooled_output)
        return scores

# 加载预训练的 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = FluencyScorer()



# 模型评分输出
model.eval()
with torch.no_grad():
    sentence = "这是一个测试句子。"
    inputs = tokenizer(sentence, return_tensors='pt')
    score = model(inputs.input_ids, inputs.attention_mask)
    print(f"这是一个测试句子。 Fluency Score: {score.item()}")

    sentence = "各个国家有各个国家的国歌"
    inputs = tokenizer(sentence, return_tensors='pt')
    score = model(inputs.input_ids, inputs.attention_mask)
    print(f"各个国家有各个国家的国歌 Fluency Score: {score.item()}")

    sentence = "各个国家有各个国家德国个"
    inputs = tokenizer(sentence, return_tensors='pt')
    score = model(inputs.input_ids, inputs.attention_mask)
    print(f"各个国家有各个国家德国个 Fluency Score: {score.item()}")
