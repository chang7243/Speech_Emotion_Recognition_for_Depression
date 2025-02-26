import os
import shutil
from pathlib import Path

def prepare_datasets(
    input_dir='/kaggle/input/splits-updated-223/splits/DAIC_WOZ',
    working_dir="/kaggle/working",
    train_dir="train",
    test_dir="test"
):
    """
    只复制训练集和测试集的音频文件到工作目录，不合并验证集
    """
    try:
        # 创建工作目录下的训练和测试目录
        working_train_dir = os.path.join(working_dir, "train")
        working_test_dir = os.path.join(working_dir, "test")
        
        # 如果目标目录已存在，先删除
        if os.path.exists(working_train_dir):
            shutil.rmtree(working_train_dir)
        if os.path.exists(working_test_dir):
            shutil.rmtree(working_test_dir)
            
        os.makedirs(working_train_dir, exist_ok=True)
        os.makedirs(working_test_dir, exist_ok=True)

        # 复制训练集音频文件
        train_files = list(Path(os.path.join(input_dir, train_dir, "audio")).rglob("*.wav"))
        print(f"找到训练集音频文件: {len(train_files)} 个")
        
        # 复制测试集音频文件
        test_files = list(Path(os.path.join(input_dir, test_dir, "audio")).rglob("*.wav"))
        print(f"找到测试集音频文件: {len(test_files)} 个")

        # 复制训练集到工作目录
        total_train_files = 0
        for file_path in train_files:
            # 直接获取文件名
            file_name = file_path.name
            target_path = os.path.join(working_train_dir, file_name)
            
            # 复制文件
            shutil.copy2(str(file_path), target_path)
            total_train_files += 1

        # 复制测试集到测试目录
        total_test_files = 0
        for file_path in test_files:
            file_name = file_path.name
            target_path = os.path.join(working_test_dir, file_name)
            shutil.copy2(str(file_path), target_path)
            total_test_files += 1

        print(f"成功复制训练集到 {working_train_dir}，共 {total_train_files} 个文件")
        print(f"成功复制测试集到 {working_test_dir}，共 {total_test_files} 个文件")
        return True

    except Exception as e:
        print(f"处理数据集时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 使用示例
    prepare_datasets() 