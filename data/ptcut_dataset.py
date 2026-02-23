import os.path
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
        
        # 获取域A图像的文件名（不含扩展名）
        A_filename = os.path.splitext(os.path.basename(A_path))[0]
        
        # 在域B中查找同名文件
        B_path = None
        for B_candidate in self.B_paths:
            B_filename = os.path.splitext(os.path.basename(B_candidate))[0]
            if A_filename == B_filename:
                B_path = B_candidate
                break
        
        # 如果找不到配对文件，使用随机选择作为备选
        if B_path is None:
            if self.opt.serial_batches:
                index_B = index % self.B_size
            else:
                index_B = random.randint(0, self.B_size - 1)
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

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
