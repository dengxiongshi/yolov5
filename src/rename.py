import os
from tqdm import tqdm
import glob


def rename_files(directory):
    # 切换到目标目录
    os.chdir(directory)

    # 获取目录下的所有文件名
    files = os.listdir()

    # 确保文件按照数字排序
    files.sort()
    pbar = tqdm(files, desc=f'Converting {directory}')  # 进度条

    i = 0
    # 重新命名文件
    for file_name in pbar:
        # 获取文件扩展名
        basename = os.path.basename(file_name)
        name = os.path.splitext(basename)[0]
        file_ext = os.path.splitext(basename)[1]

        # 构建新的文件名
        # new_name = "val_" + f"{i + 1:08d}{file_ext}"
        # new_name = "large_" + f"{i + 0:06d}.{file_ext}"
        i = i + 1
        if file_ext == '.png':
            new_name = name + '.jpg'

            # 构建完整的路径
            old_path = os.path.join(directory, file_name)
            new_path = os.path.join(directory, new_name)

            # 重命名文件
            os.rename(old_path, new_path)

        else:
            new_name = "rain_20240614_" + f"{i + 0:08d}{file_ext}"
            # new_name = "thunder_" + basename
            # 构建完整的路径
            old_path = os.path.join(directory, file_name)
            new_path = os.path.join(directory, new_name)

            # 重命名文件
            os.rename(old_path, new_path)


if __name__ == "__main__":
    # 指定目标目录
    target_directory = r"E:\downloads\compress\datasets\天气\video\rain\images"

    # 调用函数进行文件重命名
    rename_files(target_directory)
