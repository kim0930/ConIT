import torch
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

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
