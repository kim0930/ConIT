import torch
import os
from pathlib import Path

def save_model(path, model, optimizer, scheduler, epoch):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dict, path)


def dataset_path_list(root_dir):
    categories = []
    file_paths = []
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
                  file_paths += full_paths # 전체 파일 경로
                  categories_file_paths.append(full_paths)  # 카테고리별 파일 경로
                  file_label_list += [d]*len(full_paths)
                  # print([d]*len(sub_files))
                  # print(file_label_list)
    return categories, categories_file_paths, file_path_list, file_label_list
