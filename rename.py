import os


def rename_files(directory):
    # 切换到目标目录
    os.chdir(directory)

    # 获取目录下的所有文件名
    files = os.listdir()

    # 确保文件按照数字排序
    files.sort()

    # 重新命名文件
    for i, file_name in enumerate(files):
        # 获取文件扩展名
        file_ext = os.path.splitext(file_name)[1]

        # 构建新的文件名
        new_name = "test_" + f"{i + 1:08d}{file_ext}"

        # 构建完整的路径
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_path, new_path)


if __name__ == "__main__":
    # 指定目标目录
    target_directory = r"E:\downloads\compress\datasets\Car-Parts-Segmentation-master\testset\JPEGImages"

    # 调用函数进行文件重命名
    rename_files(target_directory)
