import os

def count_car_instances_for_set(file_ids, label_dir):
    difficulty_counts = {'Easy': 0, 'Moderate': 0, 'Hard': 0}

    for file_id in file_ids:
        label_file_path = os.path.join(label_dir, f"{file_id}.txt")

        if not os.path.exists(label_file_path):
            continue

        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                if not parts or parts[0] != 'Car':
                    continue

                truncated = float(parts[1])
                occluded = int(parts[2])
                bbox_top = float(parts[5])
                bbox_bottom = float(parts[7])
                
                bbox_height = bbox_bottom - bbox_top

                if bbox_height >= 40 and occluded == 0 and truncated <= 0.15:
                    difficulty_counts['Easy'] += 1
                elif bbox_height >= 25 and occluded <= 1 and truncated <= 0.30:
                    difficulty_counts['Moderate'] += 1
                elif bbox_height >= 25 and occluded <= 2 and truncated <= 0.50:
                    difficulty_counts['Hard'] += 1
    
    return difficulty_counts

def read_split_file(path):
    if not os.path.exists(path):
        print(f"エラー: スプリットファイル '{path}' が見つかりません。")
        return None
    with open(path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

if __name__ == '__main__':
    label_folder_path = 'data/kitti/training/label_2'
    train_split_path = 'data/kitti/ImageSets/myset/train.txt'
    val_split_path = 'data/kitti/ImageSets/myset/val.txt'
    test_split_path = 'data/kitti/ImageSets/myset/test.txt'

    train_ids = read_split_file(train_split_path)
    val_ids = read_split_file(val_split_path)
    test_ids = read_split_file(test_split_path)

    if train_ids is not None:
        train_counts = count_car_instances_for_set(train_ids, label_folder_path)
        train_total = sum(train_counts.values())
        
        print("="*45)
        print("📊 Training Set の 'Car' インスタンス数（重複なし）")
        print("="*45)
        print(f"📁 対象ファイル数: {len(train_ids)} 個")
        print(f"🟢 Easy:     {train_counts['Easy']:>7,} 個")
        print(f"🟡 Moderate: {train_counts['Moderate']:>7,} 個")
        print(f"🔴 Hard:     {train_counts['Hard']:>7,} 個")
        print("-"*45)
        print(f"🧮 合計:     {train_total:>7,} 個\n")

    if val_ids is not None:
        val_counts = count_car_instances_for_set(val_ids, label_folder_path)
        val_total = sum(val_counts.values())
        
        print("="*45)
        print("📊 Validation Set の 'Car' インスタンス数（重複なし）")
        print("="*45)
        print(f"📁 対象ファイル数: {len(val_ids)} 個")
        print(f"🟢 Easy:     {val_counts['Easy']:>7,} 個")
        print(f"🟡 Moderate: {val_counts['Moderate']:>7,} 個")
        print(f"🔴 Hard:     {val_counts['Hard']:>7,} 個")
        print("-"*45)
        print(f"🧮 合計:     {val_total:>7,} 個\n")

    if test_ids is not None:
        test_counts = count_car_instances_for_set(test_ids, label_folder_path)
        test_total = sum(test_counts.values())
        
        print("="*45)
        print("📊 Test Set の 'Car' インスタンス数（重複なし）")
        print("="*45)
        print(f"📁 対象ファイル数: {len(test_ids)} 個")
        print(f"🟢 Easy:     {test_counts['Easy']:>7,} 個")
        print(f"🟡 Moderate: {test_counts['Moderate']:>7,} 個")
        print(f"🔴 Hard:     {test_counts['Hard']:>7,} 個")
        print("-"*45)
        print(f"🮮 合計:     {test_total:>7,} 個")
        if test_total == 0:
            print("（注: KITTIの公式testセットにはラベルがないため、合計は通常0になります）")