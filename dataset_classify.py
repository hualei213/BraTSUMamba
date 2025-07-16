import os
import shutil

#获取不同模态的文件名
def get_unique_filenames(folder_path,destination_path):

    # 遍历文件夹中的所有文件
    for folder_name in os.listdir(folder_path):
        f_name = os.path.join(folder_path,folder_name)
        if os.path.isdir(f_name):
            parts = folder_name.split('_')
            # 检查文件名是否包含至少两个下划线
            if len(parts) >= 3:
                # 提取第二个下划线前面的部分
                name = '_'.join(parts[:2])
            else:
                print("error")

            for file_name in os.listdir(f_name):
                # filename, extension = os.path.splitext(file_name)
                parts = file_name.split('.')
                filename = parts[0]
                classify_path = os.path.join(destination_path, name, filename)
                if not os.path.exists(classify_path):
                    os.makedirs(classify_path)

def move_files_by_extension(source_folder, destination_folder):


    # 遍历源文件夹中的所有文件
    for folder_name in os.listdir(source_folder):
        f_name = os.path.join(source_folder, folder_name)
        if os.path.isdir(f_name):
            # folder_name="asl_epi_101"
            parts1 = folder_name.split('_')
            name = '_'.join(parts1[:2])
            # 获取文件夹的完整路径
            folder_path = os.path.join(source_folder, folder_name)
            # 遍历子目录
            for file_name in os.listdir(folder_path):
                parts2 = file_name.split('.')
                file_name_no_extension = parts2[0]
                destination_file = os.path.join(destination_folder, name, file_name_no_extension,
                                                parts1[2] + "_" + file_name)
                source_file = os.path.join(folder_path, file_name)
                # 移动文件到目标路径
                shutil.move(source_file, destination_file)
                print(f"Moved '{file_name}' to '{destination_file}'.")

    print("all done")


if __name__ == "__main__":
    source_path = "/mnt/HDLV1/YHR/dataset/synthstrip/synthstrip_data_v1.4"
    destination_path = "/mnt/HDLV1/YHR/dataset/synthstrip_classify"

    get_unique_filenames(source_path,destination_path)
    move_files_by_extension(source_path,destination_path)

