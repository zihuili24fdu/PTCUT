import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """
    配对图像数据集类，使用分离文件夹格式（类似CycleGAN）：
    - A和B图像分别存储在不同文件夹
    - 目录结构: /path/to/data/trainA/ 和 /path/to/data/trainB/
    - 通过文件名匹配配对图像
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # 分离的A和B文件夹
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        # 处理test阶段可能使用val文件夹的情况
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        print(f"✅ 使用分离文件夹格式")
        print(f"   A路径: {self.dir_A} ({len(self.A_paths)} 张图像)")
        print(f"   B路径: {self.dir_B} ({len(self.B_paths)} 张图像)")
        
        # 创建文件名到路径的映射，用于匹配配对
        self.B_dict = {}
        for B_path in self.B_paths:
            B_filename = os.path.splitext(os.path.basename(B_path))[0]
            self.B_dict[B_filename] = B_path
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # 通过文件名匹配配对图像
        A_path = self.A_paths[index]
        A_filename = os.path.splitext(os.path.basename(A_path))[0]
        
        # 在B中查找同名文件
        if A_filename in self.B_dict:
            B_path = self.B_dict[A_filename]
        else:
            # 如果找不到配对，抛出错误（pix2pix需要严格配对）
            raise ValueError(f"找不到与 {A_filename} 配对的B图像。"
                           f"Pix2Pix需要严格的图像配对。")
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # 对A和B应用相同的变换（保持配对关系）
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        A = A_transform(A_img)
        B = B_transform(B_img)
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

