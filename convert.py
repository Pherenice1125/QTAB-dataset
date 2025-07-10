import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import sys

def slice_image(input_path, output_dir_base, planes=['axial', 'coronal', 'sagittal']):
    """
    对指定的3D医疗影像文件进行切片，并按指定平面保存为PNG图片。

    Args:
        input_path (str): 输入的 .nii.gz 文件路径。
        output_dir_base (str): 保存切片的根目录。
        planes (list): 需要切片的方向列表, e.g., ['axial', 'coronal', 'sagittal']。
    """
    try:
        # 1. 读取图像
        sitk_image = sitk.ReadImage(input_path)
        
        # 2. 将SimpleITK图像转换为Numpy数组
        # SimpleITK 的 GetArrayFromImage 会返回一个 (depth, height, width) 顺序的数组
        # 这正好对应 (Z, Y, X) 轴，即天然的轴向(axial)切片方向
        np_image = sitk.GetArrayFromImage(sitk_image)

        # 3. 强度归一化到 0-255 以便保存为灰度图
        min_val = np.min(np_image)
        max_val = np.max(np_image)
        if max_val - min_val > 0:
            np_image = (np_image - min_val) / (max_val - min_val) * 255.0
        else:
            # 如果图像是全黑或全白的，直接设置为0
            np_image = np.zeros_like(np_image)
            
        np_image = np_image.astype(np.uint8)

        # 4. 根据指定的平面进行切片和保存
        for plane in planes:
            plane_output_dir = os.path.join(output_dir_base, plane)
            os.makedirs(plane_output_dir, exist_ok=True)

            if plane == 'axial':
                # 轴向 (Z-axis), shape[0]
                num_slices = np_image.shape[0]
                for i in range(num_slices):
                    slice_2d = np_image[i, :, :]
                    img = Image.fromarray(slice_2d)
                    img.save(os.path.join(plane_output_dir, f"slice_{i:03d}.png"))
            
            elif plane == 'coronal':
                # 冠状 (Y-axis), shape[1]
                num_slices = np_image.shape[1]
                for i in range(num_slices):
                    slice_2d = np_image[:, i, :]
                    img = Image.fromarray(slice_2d)
                    img.save(os.path.join(plane_output_dir, f"slice_{i:03d}.png"))
            
            elif plane == 'sagittal':
                # 矢状 (X-axis), shape[2]
                num_slices = np_image.shape[2]
                for i in range(num_slices):
                    slice_2d = np_image[:, :, i]
                    img = Image.fromarray(slice_2d)
                    img.save(os.path.join(plane_output_dir, f"slice_{i:03d}.png"))
        
        print(f"  -> Successfully sliced and saved to: {output_dir_base}")

    except Exception as e:
        print(f"  -> ERROR processing file {input_path}: {e}", file=sys.stderr)


def process_dataset(input_root, output_root):
    """
    遍历整个数据集目录，查找并处理FLAIR和DWI图像。

    Args:
        input_root (str): 数据集根目录 (e.g., 'ds004146').
        output_root (str): 输出切片文件的根目录 (e.g., 'slide').
    """
    if not os.path.isdir(input_root):
        print(f"Error: Input directory '{input_root}' not found.", file=sys.stderr)
        return

    print(f"Starting processing from '{input_root}'...")
    print(f"Output will be saved to '{output_root}'")
    
    # 使用os.walk遍历所有子目录
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, filename)
            
            # 检查是否是目标文件
            is_flair = 'anat' in root.split(os.sep) and filename.endswith('_FLAIR.nii.gz')
            is_dwi = 'dwi' in root.split(os.sep) and filename.endswith('_dwi.nii.gz')

            if is_flair or is_dwi:
                print(f"\nFound target file: {file_path}")

                # 1. 创建对应的输出目录结构
                # e.g., 'ds004146/sub-0001/ses-01/anat' -> 'slide/sub-0001/ses-01/anat'
                relative_path = os.path.relpath(root, input_root)
                output_path_dir = os.path.join(output_root, relative_path)
                
                # 2. 为每个图像的切片创建一个单独的文件夹
                # e.g., 'slide/.../anat/sub-0001_ses-01_FLAIR_slices'
                base_filename = filename.replace(".nii.gz", "")
                slice_output_folder = os.path.join(output_path_dir, f"{base_filename}_slices")
                
                os.makedirs(slice_output_folder, exist_ok=True)
                
                # 3. 执行切片
                slice_image(file_path, slice_output_folder)

    print("\nProcessing complete.")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 设置您的数据集根目录
    INPUT_DATASET_ROOT = 'ds004146'  # 假设脚本和数据集文件夹在同一目录下
    
    # 设置总的输出根目录
    OUTPUT_SLIDE_ROOT = 'slide'
    # --- 配置结束 ---

    process_dataset(INPUT_DATASET_ROOT, OUTPUT_SLIDE_ROOT)