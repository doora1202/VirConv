import random
import os

start_index = 0
end_index = 7480
total_files = end_index - start_index + 1


num_train = 3700
num_val = 1700
num_test = 2081

if num_train + num_val + num_test != total_files:
    print(f"エラー: 指定された分割数 ({num_train} + {num_val} + {num_test} = {num_train+num_val+num_test}) が総ファイル数 ({total_files}) と一致しません。")
else:

    file_list = [f"{i:06d}" for i in range(start_index, end_index + 1)]

    random.shuffle(file_list)

    train_files = file_list[:num_train]
    val_files = file_list[num_train : num_train + num_val]
    test_files = file_list[num_train + num_val :]

    print(f"総ファイル数: {total_files}")
    print(f"Train セットのファイル数: {len(train_files)}")
    print(f"Validation セットのファイル数: {len(val_files)}")
    print(f"Test セットのファイル数: {len(test_files)}")


    train_files_sorted = sorted(train_files, key=lambda x: int(x.split('.')[0]))
    val_files_sorted = sorted(val_files, key=lambda x: int(x.split('.')[0]))
    test_files_sorted = sorted(test_files, key=lambda x: int(x.split('.')[0]))


    with open("data/kitti/ImageSets/myset/train.txt", "w") as f:
        for file in train_files_sorted:
            f.write(file + "\n")

    with open("data/kitti/ImageSets/myset/val.txt", "w") as f:
        for file in val_files_sorted:
            f.write(file + "\n")

    with open("data/kitti/ImageSets/myset/test.txt", "w") as f:
        for file in test_files_sorted:
            f.write(file + "\n")