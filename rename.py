import os

def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if 'mixtral-8x22b-Engine' in filename:
                new_filename = filename.replace('mixtral-8x22b-Engine', 'mixtral-8x22b')
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')

if __name__ == "__main__":
    target_directory = '/data/sangchengmeng/ci_performance/H100_log_dir/lmdeploy_bf16/inference_show/log_dir/lmdeploy/Mixtral-8x22B-Engine'  # 替换为你的目标文件夹路径
    rename_files_in_directory(target_directory)
