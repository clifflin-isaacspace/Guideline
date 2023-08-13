# **視覺避障 3**

* 在這個程式中，我們將訓練我們的圖像分類器，來檢測兩個類別「free」和「blocked」，我們將用它來避免碰撞。為此，我們將使用一個熱門的深度學習庫 PyTorch。

                                    
```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

```

* 在「dataset」資料夾中收集了數據後，我們使用 `torchvision.datasets` 套件提供的 `ImageFolder` 數據集類別。我們附加來自 `torchvision.transforms` 套件的轉換，以準備訓練所需的數據。

                                    
```python
dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

```

* 接下來，我們將數據集分成訓練集和測試集。測試集將用於驗證我們訓練的模型的準確性。

                                    
```python
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
```


* 我們將創建兩個 `DataLoader` 實例，這些實例提供了對數據進行洗牌、生成圖像批次以及使用多個工作程序並行加載樣本的工具。


                                    
```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)
```


* 現在，我們定義我們將要訓練的神經網絡。`torchvision` 套件提供了一系列預訓練模型，我們可以使用它們。

* 在一個稱為轉移學習的過程中，我們可以將一個預訓練模型（在數百萬幅圖像上訓練過）用於可能具有較少數據的新任務。

* 在預訓練模型的原始訓練中學到的重要特徵對於新任務是可用的。我們將使用 `alexnet` 模型。
                                    
```python
model = models.alexnet(pretrained=True)
```

* `alexnet` 模型最初是為一個具有 1000 個類別標籤的數據集進行訓練的，但我們的數據集只有兩個類別標籤！我們將使用一個新的未訓練的最終層，該層有兩個輸出，來替換最終層。

                                    
```python
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
```

                                    
* 最後，我們將我們的模型轉移到執行階段。默認的設備是 CPU。如果你想使用 GPU 模式進行訓練，請確保你的 PyTorch 版本支援 GPU。對於 GPU 模式，`torch.device` 的參數應該是 'cuda'。
                                    
```python
device = torch.device('cpu')
model = model.to(device)
```

    
* 使用以下程式碼，我們將訓練神經網絡進行 30 個 epoch，並在每個 epoch 完成後保存表現最佳的模型。請注意，一個 epoch 是對我們的數據進行完整運行。

                         
```python
NUM_EPOCHS = 30
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):

    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy

```

* 完成後，您應該在當前工作目錄中看到一個名為 `best_model.pth` 的文件。
