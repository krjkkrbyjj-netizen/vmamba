# test_voc.py
from torchvision.datasets import VOCDetection
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

dataset = VOCDetection(
    root="/root/autodl-tmp/data",
    year='2007',
    image_set='val',
    download=False,  # 自动下载
    transform=transform
)

print("Dataset loaded successfully!")
print("Number of images:", len(dataset))

# 读取第一张图
img, target = dataset[0]
print("First image:", img.shape)
print("First target:", target)