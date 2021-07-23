import os
import torchvision.transforms as transforms

# dir
model_dir = 'Model_and_Log'
net_dir = 'ResNet'
model_name = 'MicroData'
model_path = r'D:\mengxinagjie\code\project_code\our_projects\supervised\virus_project\vgg19_bn_graph_over_classifying-130.pkl'

best_epoch = 'best_epoch'

class_num = 15
class_num2 = 30

# resnet
train_batch_size = 32
test_batch_size = 32


# GCN_model
features_dim_num = 2048
GCN_hidderlayer_dim_num = 512


# train_GCN
train_dataset_path = r'D:\datasets\Virus Texture Dataset v. 1.0\virus\train'
test_dataset_path = r'D:\datasets\Virus Texture Dataset v. 1.0\virus\test'
au_dataset_path = r'D:\datasets\Virus Texture Dataset v. 1.0\virus\train_30'

epochs = 500
k = 10

train_transform = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])











