
import torch
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import random
import signal


model = AutoModelForNextSentencePrediction.from_pretrained("snunlp/KR-BERT-char16424")
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")

#NSP 예시
sentence_A = "교수님은 나쁘다."
sentence_B = "후렌치 후라이가 아니라 벨지안 프라이야 개새끼야."
encoding = tokenizer(sentence_A, sentence_B, return_tensors='pt')
outputs = model(**encoding, labels=torch.LongTensor([1]))

logits = outputs.logits
probs = torch.softmax(log야ts, dim=1)
print(f"두 문장이 연속될 확률: {probs[0][0].item():.4f}")
print(f"두 문장이 연속되지 않을 확률: {probs[0][1].item():.4f}")

'''
tokenizer는 두 문장(sentence_A, sentence_B)을 입력받아 토큰화
return_tensors=‘pt’는 PyTorch 텐서 형식으로 출력을 반환하도록 지정
**encoding은 딕셔너리의 키-값 쌍을 개별 인자로 모델에 전달
labels=torch.LongTensor()는 이 두 문장이 연속된다고 가정하는 레이블을 제공
torch.softmax()는 로짓을 확률로 변환
'''

# 파일 읽기
with open('train_2.txt', 'r', encoding='utf-8') as file:
    content = file.read()

with open('test_2.txt', 'r', encoding='utf-8') as file:
    content_2= file.read()

# 문단 사이에 "end" 추가
modified_content = "\nend\n".join(content.split("\n\n")) + "\nend"
modified_content_2 = "\nend\n".join(content_2.split("\n\n")) + "\nend"

def preprocess_file(data):
    paragraphs = data.split('end')
    processed_paragraphs = []
    for paragraph in paragraphs:
        sentences = [sentence.strip() for sentence in paragraph.split('#@문장구분#') if sentence.strip()]
        if sentences:
            processed_paragraphs.append(sentences)
    return processed_paragraphs

train_paragraphs = preprocess_file(modified_content)
test_paragraphs= preprocess_file(modified_content_2)

def create_nsp_dataset(paragraphs, sample_ratio=0.1):
    nsp_data = []
    for sentences in paragraphs:
        # 'isnext' 쌍 생성
        for i in range(len(sentences) - 1):
            sentence_a = sentences[i]
            sentence_b = sentences[i + 1]
            nsp_data.append((sentence_a, sentence_b, 1))  # 'isnext' -> 1
        
        # 'notnext' 쌍 생성
        for i in range(len(sentences) - 1):
            sentence_a = sentences[i]
            random_paragraph = random.choice([p for p in paragraphs if p != sentences])
            sentence_b = random.choice(random_paragraph)
            nsp_data.append((sentence_a, sentence_b, 0))  # 'notnext' -> 0
    
    # 데이터셋 크기 조정 (샘플링)
    sample_size = int(len(nsp_data) * sample_ratio)
    return random.sample(nsp_data, sample_size)

# 데이터셋 크기 조정을 위해 샘플 비율을 설정
train_data = create_nsp_dataset(train_paragraphs, sample_ratio=0.1)
test_data = create_nsp_dataset(test_paragraphs, sample_ratio=0.1)


# NSPDataset 클래스 정의
class NSPDataset(Dataset):
    def __init__(self, data):
        self.sentences = data

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence_a, sentence_b, label = self.sentences[idx]
        encoding = tokenizer(sentence_a, sentence_b, return_tensors='pt', padding=True,
                             truncation=True, max_length=128)  # max_length 설정
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    token_type_ids = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.tensor([item['labels'] for item in batch])
    return input_ids, token_type_ids, attention_mask, labels

train_data = NSPDataset(train_data)
test_data = NSPDataset(test_data)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

## nvidia gpu 경우에는 "mps" 대신 "cuda"로 변경
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

class KeyboardInterruptHandler:
    def __init__(self):
        self.interrupted = False

    def handle_interrupt(self, signum, frame):
        self.interrupted = True
        torch.save(model.state_dict(), 'interrupted_model.pth')

handler = KeyboardInterruptHandler()
signal.signal(signal.SIGINT, handler.handle_interrupt)

epochs = 3
losses = []

model.train()
try:
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            input_ids, token_type_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} completed with average loss: {avg_loss}")

except KeyboardInterrupt:
    print("Training was interrupted by user.")

# 손실 시각화
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

#모델 평가

model.eval()

correct_predictions = 0
total_predictions = 0


with torch.no_grad():
    for batch in test_loader:
        input_ids, token_type_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)


accuracy = correct_predictions / total_predictions
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
