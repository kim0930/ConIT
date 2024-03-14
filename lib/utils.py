import torch
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset, Sampler

def save_model(path, model, optimizer, scheduler, epoch):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dict, path)


def prepare_dataset(root_dir, split_ratio):
    '''
    split_ratio: [train, val, test] / For example, [0.8, 0.1, 0.1] / val=test는 항상 같게. 
    
    ''' 
    categories = []
    file_path_list = []
    file_label_list = []
    categories_file_paths = []
    path = os.getcwd()
    for (root, dirs, files) in os.walk(root_dir):
        for d in dirs:
            if not d.startswith('.'):
                fp = Path(root) / d
                # print(fp)
                categories.append(d)
                for (sub_root, sub_dirs, sub_files) in os.walk(fp):
                  # print(sub_files)
                  full_paths = []
                  for s_path in sub_files:
                    full_paths.append(str(path / fp) + "/" + s_path)
                  file_path_list += full_paths # 전체 파일 경로
                  categories_file_paths.append(full_paths)  # 카테고리별 파일 경로
                  file_label_list += [d]*len(full_paths)
                  # print([d]*len(sub_files))
                  # print(file_label_list)


    df = pd.DataFrame(file_path_list, columns=['Path'])
    df['Label'] = file_label_list
    df.to_csv("dataset.csv", index=False)
    pipe = pd.read_csv('./dataset.csv')
    # print(pipe.head())
    # print(pipe.tail())
    
    # 학습을 위해 라벨을 재할당
    mapping = {}
    for i in range(len(categories)):
      mapping[categories[i]] = i
    pipe.Label = pipe.Label.map(mapping)
    
    np.random.seed(42)  # random seed 고정 (결과 재생산 가능하도록)
    random_index = np.random.permutation(pipe.index) # 랜덤하게 인덱스 생성
    pipe = pipe.iloc[random_index] # 랜덤하게 생성한 인덱스 순서로 재배열
    pipe = pipe.reset_index(drop=True) # 다시 순서대로 인덱스 부여
    
    print(pipe.head(10))
    print()

    data_frame = pipe
    train_size = int(split_ratio[0] * len(data_frame))  # 80%를 train, 20%를 validation으로 사용
    val_size = int((len(data_frame) - train_size)/2)
    test_size = len(data_frame) - train_size - val_size
    
    print(f"Total:{len(data_frame)}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    train_data = data_frame[:train_size]
    val_data = data_frame[train_size:train_size + val_size]
    test_data = data_frame[train_size + val_size: ]
    
    for i in range(len(categories)):
      tra = len(train_data[train_data["Label"]==i])
      va =len(val_data[val_data["Label"]==i])
      tes = len(test_data[test_data["Label"]==i])
    
      print(f"- Category: {categories[i]}, Total:{tra+va+tes}, Train: {tra}, Val: {va}, Test: {tes}")
    
    
    # return categories, categories_file_paths, file_path_list, file_label_list
    return train_data, val_data, test_data

def visualize_img(dataset, num):
    '''
    num: 보여줄 이미지의 수
    '''
    # 데이터프레임에서 이미지 파일 경로 및 라벨 추출
    data_image_paths = (dataset["Path"]).tolist()
    data_labels = dataset["Label"].tolist()
    
    i=1
    for path in data_image_paths:
      img = Image.open(path)
      plt.subplot(2, int(num/2), i)
      plt.imshow(img)
      print(f"{path}")
      plt.xticks([])
      plt.yticks([])
      i+=1
      if i > num:
        break
    plt.tight_layout()
    plt.show()

class CustomDataset(Dataset):
    def __init__(self, data_image_paths, data_labels, transform=None):
        self.data_image_paths = data_image_paths
        self.data_labels = data_labels
        self.transform = transform

    def __len__(self):
        return len(self.data_image_paths)

    def __getitem__(self, idx):
        img_path = self.data_image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.data_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# val_dataset에서 각각 라벨이 0,1,2,3 인 데이터를 4개씩 랜덤 추출


class LabelSampler(Sampler): # 라벨별로 지정된 수만큼 데이터를 추출
    def __init__(self, dataset, num_samples_per_label):
        self.dataset = dataset
        self.num_samples_per_label = num_samples_per_label
        self.label_indices = self._create_label_indices()

    # 각 라벨에 해당하는 인덱스를 모아 딕셔너리로 저장
    def _create_label_indices(self):
        label_indices = {label: [] for label in set(self.dataset.data_labels)}
        for i, label in enumerate(self.dataset.data_labels):
            label_indices[label].append(i)
        return label_indices

    # 각 라벨에 대해 지정된 수만큼 무작위로 인덱스를 선택하여 반환
    def __iter__(self):
        indices = []
        for label, indices_per_label in self.label_indices.items():
            if len(indices_per_label) >= self.num_samples_per_label:
                selected_indices = torch.randperm(len(indices_per_label))[:self.num_samples_per_label]
                indices.extend([indices_per_label[idx] for idx in selected_indices])
            else:
                indices.extend(indices_per_label * self.num_samples_per_label)
        return iter(indices)

    # 데이터셋의 전체 길이를 반환
    def __len__(self):
        return len(self.dataset)


# 이미지의 RGB 채널별 통계량 확인 함수
def normalize_dataset(data1, datas2):
    # Transform and Load Data
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
        ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
        ])
    # Update the datasets with the new transform
    dataset1 = CustomDataset(data1["Path"], data1["Label"], transform = train_transform)
    dataset2 = CustomDataset(data2["Path"], data2["Label"], transform = test_transform)
    dataset = dataset1 + dataset2
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()

    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')
    return  [mean_r, mean_g, mean_b], [std_r, std_g, std_b]
