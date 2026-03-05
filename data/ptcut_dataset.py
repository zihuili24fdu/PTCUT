import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class PtcutDataset(BaseDataset):
    """
    PTCUT专用数据集类 - 支持配对图像且A和B共享裁剪参数
    
    特点：
    1. 通过文件名匹配A和B域的配对图像
    2. A和B使用相同的随机裁剪位置（像素级对齐）
    3. 适用于需要语义监督的PTCUT模型
    
    目录结构:
    /path/to/data/trainA/ 和 /path/to/data/trainB/
    /path/to/data/testA/ 和 /path/to/data/testB/
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # 预构建 filename -> path 字典，O(1) 查找，避免每次 __getitem__ 都线性扫描
        self.B_dict = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in self.B_paths
        }

        print(f"✅ [PTCUT Dataset] 使用共享裁剪参数的配对数据集")
        print(f"   A路径: {self.dir_A} ({self.A_size} 张图像)")
        print(f"   B路径: {self.dir_B} ({self.B_size} 张图像)")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]

        # O(1) 字典查找配对 B 图像
        A_filename = os.path.splitext(os.path.basename(A_path))[0]
        if A_filename in self.B_dict:
            B_path = self.B_dict[A_filename]
        else:
            # 找不到配对文件时随机回退
            index_B = index % self.B_size if self.opt.serial_batches else random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation with shared parameters
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        # 🔑 关键：生成共享的transform参数，确保A和B使用相同的裁剪位置
        transform_params = get_params(modified_opt, A_img.size)
        A_transform = get_transform(modified_opt, transform_params)
        B_transform = get_transform(modified_opt, transform_params)
        
        A = A_transform(A_img)
        B = B_transform(B_img)

        # 在 DataLoader worker 中预解析标签，避免在模型前向中解析路径字符串
        A_label = self._parse_label(A_path)
        B_label = self._parse_label(B_path)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
                'A_label': A_label, 'B_label': B_label}

    @staticmethod
    def _parse_label(image_path):
        """从文件名中解析类别标签，返回 LongTensor scalar。

        支持格式:
          新格式: *_i.jpg -> 0 (intermixed/composite), *_n.jpg -> 1 (nodular)
          旧格式: *_1.jpg -> 0, *_2.jpg -> 1, *_3.jpg -> 2, *_4.jpg -> 3
        """
        label_map = {
            'i': 0,  # intermixed / composite
            'n': 1,  # nodular
            '1': 0, '2': 1, '3': 2, '4': 3,  # 旧版数字格式
        }
        filename = os.path.splitext(os.path.basename(image_path))[0]
        suffix = filename.split('_')[-1]  # 取最后一个 '_' 后的部分
        label = label_map.get(suffix, 0)  # 未知后缀默认 0
        return torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
