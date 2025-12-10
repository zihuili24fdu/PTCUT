"""
文本生成器模块
用于从txt文档读取病理组织细胞HE图像的文本描述
"""

import random
import os
from typing import List


class TextGenerator:
    """
    文本描述生成器类 - 从txt文件读取病理描述
    """
    
    def __init__(self, text_file_path):
        """
        初始化文本生成器
        
        Args:
            text_file_path: str, 包含病理描述的txt文件路径
        """
        self.text_file_path = text_file_path
        self.file_descriptions = []
        
        # 加载文件描述
        self._load_descriptions_from_file()
        
    def _load_descriptions_from_file(self):
        """
        从文件加载文本描述
        支持多种文件格式：
        - 每行一个描述
        - 空行分隔的多行描述
        - 以分号分隔的描述
        """
        if not self.text_file_path:
            raise ValueError("必须指定文本文件路径")
            
        if not os.path.exists(self.text_file_path):
            raise FileNotFoundError(f"文本文件不存在: {self.text_file_path}")
            
        try:
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                raise ValueError(f"文本文件为空: {self.text_file_path}")
                
            # 尝试不同的分割方式
            descriptions = []
            
            # 方法1: 以双换行符分割（支持多行描述）
            if '\n\n' in content:
                descriptions = [desc.strip().replace('\n', ' ') for desc in content.split('\n\n') 
                              if desc.strip() and not desc.strip().startswith('#')]
            
            # 方法2: 以单换行符分割（每行一个描述）
            elif '\n' in content:
                descriptions = [line.strip() for line in content.split('\n') 
                              if line.strip() and not line.strip().startswith('#')]
            
            # 方法3: 以分号分割
            elif ';' in content:
                descriptions = [desc.strip() for desc in content.split(';') 
                              if desc.strip() and not desc.strip().startswith('#')]
            
            # 方法4: 整个文件作为一个描述
            else:
                descriptions = [content] if not content.strip().startswith('#') else []
                
            # 过滤过短的描述
            descriptions = [desc for desc in descriptions if len(desc) > 20]
            
            if not descriptions:
                raise ValueError(f"未找到有效的文本描述（要求长度>20字符）")
                
            self.file_descriptions = descriptions
            print(f"成功加载 {len(self.file_descriptions)} 个文本描述，来自文件: {self.text_file_path}")
            
            # 显示前3个描述的预览
            print("文本描述预览:")
            for i, desc in enumerate(self.file_descriptions[:3], 1):
                preview = desc[:80] + '...' if len(desc) > 80 else desc
                print(f"  {i}. {preview}")
            if len(self.file_descriptions) > 3:
                print(f"  ... 还有 {len(self.file_descriptions) - 3} 个描述")
                
        except Exception as e:
            raise RuntimeError(f"读取文本文件失败: {e}")
            
    def reload_descriptions_from_file(self, new_file_path=None):
        """
        重新加载文件描述（用于动态更新）
        
        Args:
            new_file_path: str, 新的文件路径（可选）
        """
        if new_file_path:
            self.text_file_path = new_file_path
        self._load_descriptions_from_file()
        
    def get_single_description(self) -> str:
        """
        从文件描述中随机选择一个
        
        Returns:
            str: 随机选择的病理描述
        """
        if not self.file_descriptions:
            raise RuntimeError("没有可用的文本描述")
            
        return random.choice(self.file_descriptions)

    def get_batch_descriptions(self, batch_size: int) -> List[str]:
        """
        批量获取描述
        
        Args:
            batch_size: int, 需要的描述数量
            
        Returns:
            List[str]: 描述列表
        """
        if batch_size <= 0:
            return []
            
        if not self.file_descriptions:
            raise RuntimeError("没有可用的文本描述")
            
        # 如果需要的数量小于等于可用描述数量，随机选择不重复的描述
        if batch_size <= len(self.file_descriptions):
            return random.sample(self.file_descriptions, batch_size)
        
        # 如果需要的数量大于可用描述数量，允许重复选择
        descriptions = []
        for _ in range(batch_size):
            descriptions.append(random.choice(self.file_descriptions))
        
        return descriptions

    def get_all_descriptions(self) -> List[str]:
        """
        获取所有描述
        
        Returns:
            List[str]: 所有描述的列表
        """
        return self.file_descriptions.copy()
    
    def get_descriptions_count(self) -> int:
        """
        获取描述数量
        
        Returns:
            int: 描述总数
        """
        return len(self.file_descriptions)
    
    def __call__(self, batch_size: int = 1) -> List[str]:
        """
        调用接口
        
        Args:
            batch_size: int, 批大小
            
        Returns:
            List[str]: 描述列表
        """
        return self.get_batch_descriptions(batch_size)

    def __len__(self) -> int:
        """
        返回描述数量
        """
        return len(self.file_descriptions)
    
    def __str__(self) -> str:
        """
        字符串表示
        """
        return f"TextGenerator(file='{self.text_file_path}', descriptions={len(self.file_descriptions)})"