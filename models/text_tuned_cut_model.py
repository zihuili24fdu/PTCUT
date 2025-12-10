import torch
import torch.nn.functional as F
from .cut_model import CUTModel
from util.text_generator import TextGenerator
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


class TextTunedCUTModel(CUTModel):
    """
    文本调优的CUT模型，用于虚拟染色任务
    在CUT模型基础上增加文本调优功能，包含：
    1. 使用大语言模型生成病理组织细胞HE图像描述
    2. 使用CONCH模型提取图像和文本的embedding
    3. 计算文本-图像相似度损失(loss1)和图像embedding损失(loss2)
    
    多GPU训练优化特性：
    - CONCH模型支持DataParallel并行推理
    - 文本embeddings预计算并缓存在CPU，按需移到GPU
    - 批量处理图像编码，减少GPU间数据传输
    - 智能设备管理，确保所有tensor在正确的设备上
    - 内存优化，支持大batch_size训练
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ 配置文本调优CUT模型特有的参数选项 """
        parser = CUTModel.modify_commandline_options(parser, is_train)
        
        # 文本调优相关参数
        parser.add_argument('--lambda_text', type=float, default=1, help='文本相似度损失的权重')
        parser.add_argument('--lambda_feat', type=float, default=1, help='特征相似度损失的权重')
        # 测试模式下不需要文本描述文件
        parser.add_argument('--text_descriptions_file', type=str, 
                           required=is_train,  # 仅训练时必需
                           default='',
                           help='病理组织HE图像文本描述文件路径（训练时必需）')
        return parser

    def __init__(self, opt):
        super(TextTunedCUTModel, self).__init__(opt)
        
        # 标记模型尚未完全初始化
        self._initialized = False
        
        # 保存选项
        self.opt = opt
        
        # 检查是否在训练模式
        self.isTrain = opt.isTrain
        
        # 测试模式优化：跳过CONCH和文本相关的初始化
        if not self.isTrain:
            print("🚀 测试模式：跳过CONCH模型和文本描述加载，节省内存...")
            self.conch_model = None
            self.conch_device = None
            self.text_generator = None
            self.text_embeddings = None
            self.text_descriptions = []
            # 测试模式不需要这些损失
            return
            
        # ============ 以下仅在训练模式执行 ============
        print("🔧 训练模式：初始化CONCH模型和文本描述...")
        
        # 初始化CONCH模型包装器（在主GPU上）
        # 获取主设备（第一个GPU或CPU）
        self.conch_device = torch.device(f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) > 0 else 'cpu')
        
        self.conch_model, self.conch_preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", 
            checkpoint_path="checkpoints/conch/pytorch_model.bin"
            )
        self.conch_model.eval()
        self.conch_model.to(self.conch_device)
        
        # 多GPU优化：如果有多个GPU，使用DataParallel包装CONCH模型
        if len(opt.gpu_ids) > 1:
            self.conch_model = torch.nn.DataParallel(self.conch_model, device_ids=opt.gpu_ids)
            print(f"CONCH模型已启用DataParallel，使用GPU: {opt.gpu_ids}")
        
        # 初始化文本生成器
        if not hasattr(opt, 'text_descriptions_file') or not opt.text_descriptions_file:
            raise ValueError("必须指定文本描述文件路径 --text_descriptions_file")
            
        self.text_generator = TextGenerator(text_file_path=opt.text_descriptions_file)
            
        # 添加新的损失名称
        self.loss_names.extend(['TEXT', 'FEAT'])
        
        # 文本描述缓存
        self.text_descriptions = []
        self.text_embeddings = None  # 初始化为None，稍后加载
        
        # 一次性加载并编码所有文本描述
        self._load_and_encode_all_texts()

    def _load_and_encode_all_texts(self):
        """
        一次性加载并编码所有文本描述（最大化性能优化）
        多GPU优化：文本embeddings只在主GPU上计算一次，然后固定住
        """
        # 加载文件中的所有描述
        all_descriptions = self.text_generator.get_all_descriptions()
        print(f"从文件加载了 {len(all_descriptions)} 个文本描述")
        
        # 一次性编码所有描述，训练中不再重复计算
        tokenizer = get_tokenizer()
        text_emb_list = []
        
        # 批量处理文本以提高效率
        batch_size = 16  # 可以根据GPU内存调整
        for i in range(0, len(all_descriptions), batch_size):
            batch_texts = all_descriptions[i:i+batch_size]
            text_tokens = tokenize(texts=batch_texts, tokenizer=tokenizer).to(self.conch_device)
            
            with torch.inference_mode():
                # 处理DataParallel包装的模型
                if isinstance(self.conch_model, torch.nn.DataParallel):
                    text_emb = self.conch_model.module.encode_text(text_tokens, normalize=True)
                else:
                    text_emb = self.conch_model.encode_text(text_tokens, normalize=True)
            
            # 移到CPU以节省GPU内存，训练时再移到对应设备
            text_emb_list.append(text_emb.cpu())
        
        # stack 后形状: [N, 512]，存储在CPU上
        self.text_embeddings = torch.cat(text_emb_list, dim=0)
        print(f"文本编码完成，形状: {self.text_embeddings.shape}，存储在CPU，训练中将按需移到GPU")

    def compute_text_similarity_loss(self, generated_images, text_embeddings):
        """
        计算生成图像与文本描述的相似度损失 (loss1)
        采用对比学习思想，使生成图像的embedding与文本描述的embedding更接近
        多GPU优化：确保所有tensor在同一设备上
        """
        # 测试模式或无效输入时返回零损失
        if not self.isTrain or self.conch_model is None:
            return torch.tensor(0.0, requires_grad=True, device=generated_images.device)
        
        if text_embeddings is None or text_embeddings.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=generated_images.device)

        # 获取当前batch的设备
        current_device = generated_images.device
        
        # 将文本embeddings移到当前设备（如果还在CPU上）
        text_embeddings = text_embeddings.to(current_device)

        # 预处理生成图像：Tensor (B, C, H, W) 值域[-1,1] -> 调整大小并归一化
        # 1. 将值域从 [-1, 1] 转换到 [0, 1]
        imgs = (generated_images + 1) / 2.0
        # 2. 调整到 CONCH 期望的输入尺寸 (512x512)
        imgs = F.interpolate(imgs, size=(512, 512), mode='bilinear', align_corners=False)
        # 3. 归一化（CONCH 使用 ImageNet 统计值）
        mean = torch.tensor([0.485, 0.456, 0.406], device=current_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=current_device).view(1, 3, 1, 1)
        image_tensor = (imgs - mean) / std

        # 编码生成图像（使用inference_mode而非no_grad以提高性能）
        with torch.inference_mode():
            # 将图像移到CONCH设备进行推理
            image_tensor_conch = image_tensor.to(self.conch_device)
            
            # 处理DataParallel包装的模型
            if isinstance(self.conch_model, torch.nn.DataParallel):
                gen_img_embeddings = self.conch_model.module.encode_image(image_tensor_conch, normalize=True)
            else:
                gen_img_embeddings = self.conch_model.encode_image(image_tensor_conch, normalize=True)
            
            # 移回当前设备
            gen_img_embeddings = gen_img_embeddings.to(current_device)

        # 计算相似度矩阵 (B, N)
        # gen_img_embeddings: [B, 512], text_embeddings: [N, 512]
        # 结果: [B, N]
        similarity = torch.matmul(
            F.normalize(gen_img_embeddings, dim=1),
            F.normalize(text_embeddings, dim=1).transpose(0, 1)  # [N, 512] -> [512, N]
        )

        # 损失：每个生成图像与所有文本描述的最大相似度
        max_similarities, _ = similarity.max(dim=1)
        loss = 1 - max_similarities.mean()

        return loss

    def compute_feature_similarity_loss(self, real_images, generated_images):
        """
        计算真实图像与生成图像的embedding损失 (loss2)
        使生成图像的特征与真实图像的特征保持一致
        多GPU优化：批量处理图像编码，减少设备间传输
        """
        # 测试模式时返回零损失
        if not self.isTrain or self.conch_model is None:
            return torch.tensor(0.0, requires_grad=True, device=generated_images.device)
        
        # 获取当前batch的设备
        current_device = generated_images.device
        
        # 预处理函数：Tensor预处理
        def preprocess_tensor(imgs):
            # 1. 将值域从 [-1, 1] 转换到 [0, 1]
            imgs = (imgs + 1) / 2.0
            # 2. 调整到 CONCH 期望的输入尺寸 (512x512)
            imgs = F.interpolate(imgs, size=(512, 512), mode='bilinear', align_corners=False)
            # 3. 归一化（CONCH 使用 ImageNet 统计值）
            mean = torch.tensor([0.485, 0.456, 0.406], device=current_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=current_device).view(1, 3, 1, 1)
            return (imgs - mean) / std
        
        real_images_tensor = preprocess_tensor(real_images)
        generated_images_tensor = preprocess_tensor(generated_images)
        
        # 批量编码以减少开销
        with torch.inference_mode():
            # 合并真实图像和生成图像进行批量处理
            combined_images = torch.cat([real_images_tensor, generated_images_tensor], dim=0)
            combined_images = combined_images.to(self.conch_device)
            
            # 处理DataParallel包装的模型
            if isinstance(self.conch_model, torch.nn.DataParallel):
                combined_embeddings = self.conch_model.module.encode_image(combined_images, normalize=True)
            else:
                combined_embeddings = self.conch_model.encode_image(combined_images, normalize=True)
            
            # 移回当前设备并分离真实和生成图像的embeddings
            combined_embeddings = combined_embeddings.to(current_device)
            batch_size = real_images.shape[0]
            real_img_embeddings = combined_embeddings[:batch_size]
            gen_img_embeddings = combined_embeddings[batch_size:]

        # 计算余弦相似度损失
        similarity = torch.cosine_similarity(real_img_embeddings, gen_img_embeddings, dim=1)
        loss = 1 - similarity.mean()

        return loss

    def data_dependent_initialize(self, data):
        """重写父类方法，在初始化完成后标记"""
        super().data_dependent_initialize(data)
        self._initialized = True
    
    def forward(self):
        """前向传播；被<optimize_parameters>和<test>调用。"""
        # 调用父类的前向传播
        super().forward()

    def compute_G_loss(self):
        """计算生成器的GAN、NCE和文本调优损失"""
        # 调用父类计算原始损失
        loss_G = super().compute_G_loss()
        
        # 测试模式：只返回父类损失，不计算文本调优损失
        if not self.isTrain:
            self.loss_G = loss_G
            return self.loss_G
        
        # 在初始化阶段跳过文本调优损失，避免卡住
        if not hasattr(self, '_initialized') or not self._initialized:
            self.loss_TEXT = torch.tensor(0.0, requires_grad=True).to(self.device)
            self.loss_FEAT = torch.tensor(0.0, requires_grad=True).to(self.device)
            self.loss_G = loss_G
            return self.loss_G
        
        # 计算文本相似度损失
        if getattr(self.opt, 'lambda_text', 0.0) > 0.0:
            self.loss_TEXT = self.compute_text_similarity_loss(
                self.fake_B, self.text_embeddings
            ) * self.opt.lambda_text
        else:
            self.loss_TEXT = torch.tensor(0.0, requires_grad=True).to(self.device)
            
        # 计算特征相似度损失
        if getattr(self.opt, 'lambda_feat', 0.0) > 0.0:
            self.loss_FEAT = self.compute_feature_similarity_loss(
                self.real_B, self.fake_B
            ) * self.opt.lambda_feat
        else:
            self.loss_FEAT = torch.tensor(0.0, requires_grad=True).to(self.device)
            
        # 总损失
        self.loss_G = loss_G + self.loss_TEXT + self.loss_FEAT
        return self.loss_G

    def get_current_visuals(self):
        """返回当前的可视化结果"""
        visual_ret = super().get_current_visuals()
        
        # 添加文本描述到可视化信息中
        if hasattr(self, 'text_descriptions') and self.text_descriptions:
            visual_ret['text_descriptions'] = self.text_descriptions[:4]  # 只显示前4个
            
        return visual_ret
    
    def optimize_memory(self):
        """
        优化GPU内存使用
        在训练过程中定期调用可以释放未使用的缓存
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_model_info(self):
        """
        返回模型信息，用于调试和监控
        """
        info = {
            'mode': 'train' if self.isTrain else 'test',
            'num_gpus': len(self.opt.gpu_ids),
            'conch_device': str(self.conch_device) if self.conch_device is not None else 'None (test mode)',
            'conch_is_parallel': isinstance(self.conch_model, torch.nn.DataParallel) if self.conch_model is not None else False,
            'text_embeddings_shape': self.text_embeddings.shape if self.text_embeddings is not None else None,
            'lambda_text': getattr(self.opt, 'lambda_text', 0.0),
            'lambda_feat': getattr(self.opt, 'lambda_feat', 0.0),
        }
        return info