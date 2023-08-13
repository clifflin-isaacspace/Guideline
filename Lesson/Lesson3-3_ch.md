# **Road Following 3**

* 在這個腳本中，我們將訓練一個神經網絡，該神經網絡將輸入一個圖像，並輸出一組對應於目標的 x 和 y 值。

* 我們將使用 PyTorch 深度學習框架，訓練一個 ResNet-18 神經網絡架構模型，用於道路跟隨應用。

                                    
```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

```

## 建立資料集Instance
***

* 在這裡，我們創建了一個自定義的 `torch.utils.data.Dataset` 實現，該實現包括了 `__len__` 和 `__getitem__` 函數。這個類負責加載圖像並從圖像檔名中解析出 x 和 y 值。由於我們實現了 `torch.utils.data.Dataset` 類，我們可以使用所有的 torch 資料工具。

* 我們在資料集中硬編碼了一些轉換（如顏色抖動）。我們將隨機水平翻轉作為選項（以防需要遵循非對稱的路徑，比如需要“靠右”的道路）。如果你的機器人是否遵循某種慣例不重要，你可以啟用翻轉來增加數據集的多樣性。

* 從圖像檔名中獲取 x 和 y 值。
                                    
```python
def get_x(path, width):
    return (float(int(path.split("_")[1])) - width/2) / (width/2)

def get_y(path, height):
    return (float(int(path.split("_")[2])) - height/2) / (height/2)
```

* 建立讀取Dataset的class

```python

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        width, height = image.size
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))
      
        if (float(np.random.rand(1)) > 0.5) and self.random_hflips:
            image = transforms.functional.hflip(image)
            x = -x
        
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor([x, y]).float()
    
dataset = XYDataset('dataset_xy', random_hflips=False)

```

                                    
## 資料集分割為訓練集和測試集
***

* 一旦我們讀取了資料集，我們將把資料集分成訓練集和測試集。在這個例子中，我們將資料集分成90%的訓練集和10%的測試集。測試集將用於驗證我們所訓練模型的準確性。

                           
```python
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

```

## 建立數據加載器以分批加載數據
***

* 我們使用 `DataLoader` 類來批量加載數據，對數據進行洗牌並允許使用多個子進程。在這個例子中，我們使用批量大小為 8。批量大小將基於您的 GPU/RAM 可用內存，並且可能會影響模型的準確性。

                           
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

## 定義神經網絡模型
***

* 我們使用 PyTorch TorchVision 中提供的 [ResNet-18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) 模型。

* 在一個稱為 [遷移學習](https://www.youtube.com/watch?v=yofjFQddwHE) 的過程中，我們可以將一個已經在數百萬張圖像上進行過訓練的預訓練模型重新應用到一個新的任務，即使這個新任務可能具有更少的可用數據。                               

```python
model = models.resnet18(pretrained=True)
```

                                    
* ResNet 模型具有全連接（fc）的最後一層，其 `in_features` 為 512，我們將進行 x 和 y 的回歸訓練，因此 `out_features` 設為 2。

* 最後，我們將模型轉移到 CPU 上進行執行。如果您已經安裝了支援 GPU 的 PyTorch 版本，最好在 GPU 上執行模型。
                     
```python
model.fc = torch.nn.Linear(512, 2)
device = torch.device('cpu') # input 'cuda' for using GPU
model = model.to(device)


```

## 訓練回歸模型
***

* 我們進行 50 個訓練周期，如果損失減少，就保存最佳模型。

                                    
```python
NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_steering_model_xy.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)
    
    print('%f, %f' % (train_loss, test_loss))
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss

```

* 一旦模型訓練完成，它將生成 `best_steering_model_xy.pth` 檔案，您可以在實時演示的 Python 腳本中進行推論。
