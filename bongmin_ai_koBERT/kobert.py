import json
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from kobert import get_mxnet_kobert_model, get_tokenizer
from mxnet.gluon import nn
from mxnet import gluon
from tqdm import tqdm

# KoBERT 모델과 토크나이저 로딩
ctx = mx.cpu()  # GPU 사용 시, mx.gpu()로 변경 가능

bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx, cachedir=".cache")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# BERTDataset 클래스 정의
class BERTDataset(mx.gluon.data.Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad=True, pair=False):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        sent_dataset = gluon.data.SimpleDataset([[i[sent_idx]] for i in dataset])
        self.sentences = sent_dataset.transform(transform)
        self.labels = gluon.data.SimpleDataset([np.array(np.int32(i[label_idx])) for i in dataset])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return len(self.labels)

# 업로드한 JSON 파일 읽기 (낚시성 기사와 비낚시성 기사)
with open('/mnt/data/ET_M03_165162_L.json') as f:
    clickbait_data = json.load(f)
with open('/mnt/data/ET_M03_276098_L.json') as f:
    non_clickbait_data = json.load(f)

# 데이터 처리 및 (뉴스 내용, 레이블) 리스트 만들기
train_data = []
test_data = []

# clickbait_data에서 뉴스 내용과 레이블 추출
for item in clickbait_data:
    train_data.append([item["sourceDataInfo"]["newsContent"], 1])  # 클릭베이트는 레이블 1
for item in non_clickbait_data:
    train_data.append([item["sourceDataInfo"]["newsContent"], 0])  # 비클릭베이트는 레이블 0

# train_data와 test_data를 분리하거나, 필요한 데이터를 넣으시면 됩니다.
max_len = 128  # 최대 문장 길이 설정

# 훈련 데이터셋과 테스트 데이터셋 생성
data_train = BERTDataset(train_data, 0, 1, tok, max_len, True, False)

# 모델 정의
class BERTClassifier(nn.Block):
    def __init__(self, bert, num_classes=2, dropout=None, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):
        _, pooler = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler)

# 모델 초기화
model = BERTClassifier(bert_base, num_classes=2, dropout=0.1)
model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
model.hybridize()

# 학습에 사용할 손실 함수와 평가 지표 정의
loss_function = gluon.loss.SoftmaxCELoss()
metric = mx.metric.Accuracy()

# 훈련과 테스트용 배치 크기 설정
batch_size = 32
lr = 5e-5

train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)

trainer = gluon.Trainer(model.collect_params(), 'bertadam', {'learning_rate': lr, 'epsilon': 1e-9, 'wd': 0.01})

# 학습
num_epochs = 5
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        with mx.autograd.record():
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # 모델 학습
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # 역전파
        ls.backward()
        trainer.update(1)
        step_loss += ls.asscalar()
        metric.update([label], [out])

    print(f'Epoch {epoch_id + 1}, Loss: {step_loss / len(train_dataloader)}, Accuracy: {metric.get()[1]}')

# 학습 완료 후 평가
def evaluate_accuracy(model, data_iter, ctx=ctx):
    acc = mx.metric.Accuracy()
    for i, (t, v, s, label) in enumerate(data_iter):
        token_ids = t.as_in_context(ctx)
        valid_length = v.as_in_context(ctx)
        segment_ids = s.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(token_ids, segment_ids, valid_length.astype('float32'))
        acc.update(preds=output, labels=label)
    return acc.get()[1]

# 평가 수행 (테스트 데이터셋)
test_acc = evaluate_accuracy(model, test_dataloader, ctx)
print(f'Test Accuracy: {test_acc}')
