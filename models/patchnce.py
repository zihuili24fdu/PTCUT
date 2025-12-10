from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # 根据PyTorch版本选择mask的dtype
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # 正样本logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # 负样本logit

        # 是否利用minibatch中其他样本的负样本？
        # 在CUT和FastCUT中，发现只包含同一张图片的负样本效果最好，因此
        # --nce_includes_all_negatives_from_minibatch 默认为False
        # 但对于单张图片的翻译，minibatch由同一高分辨率图片的不同crop组成，
        # 因此会包含整个minibatch的负样本
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # 将特征重塑为仿佛minibatch为1时的所有负样本
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # 将特征reshape为batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # 对角线元素是同一特征的相似度，没有意义
        # 用很小的数（exp(-10)，几乎为零）填充对角线
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
