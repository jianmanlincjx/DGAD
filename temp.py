# import torch
# import torch.nn as nn

# # 假设 CrossAttentionFusion 已经定义好，如前面的代码示例
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, channels, num_heads=8, dim_head=64):
#         super().__init__()
#         inner_dim = num_heads * dim_head
#         self.to_q = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
#         self.to_k = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
#         self.to_v = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
#         self.mha = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=num_heads, batch_first=True)
#         self.to_out = nn.Conv2d(inner_dim, channels, kernel_size=1)

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape
#         q = self.to_q(feat1)
#         k = self.to_k(feat2)
#         v = self.to_v(feat2)
#         q = q.flatten(2).transpose(1, 2)
#         k = k.flatten(2).transpose(1, 2)
#         v = v.flatten(2).transpose(1, 2)
#         out, _ = self.mha(q, k, v)
#         out = out.transpose(1, 2).view(B, -1, H, W)
#         return self.to_out(out)

# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.first = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
#         # down block
#         self.down_0_0 = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
#         self.down_0_1 = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
#         self.down_0_1_downsamplers = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
        
#         self.down_1_0 = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
#         self.down_1_1 = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
#         self.down_1_1_downsamplers = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
        
#         self.down_2_0 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.down_2_1 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.down_2_1_downsamplers = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
        
#         self.down_3_0_resnet = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.down_3_1_resnet = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
        
#         # mid block
#         self.mid_block_add_sample = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
        
#         # up block
#         self.up_0_0 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_0_1 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_0_2 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_0_2_upsamplers = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
        
#         self.up_1_0 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_1_1 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_1_2 = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
#         self.up_1_2_upsamplers = CrossAttentionFusion(channels=1280, num_heads=8, dim_head=40)
        
#         self.up_2_0 = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
#         self.up_2_1 = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
#         self.up_2_2 = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
#         self.up_2_2_upsamplers = CrossAttentionFusion(channels=640, num_heads=8, dim_head=40)
        
#         self.up_3_0 = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
#         self.up_3_1 = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)
#         self.up_3_2 = CrossAttentionFusion(channels=320, num_heads=8, dim_head=40)

#     def forward(self, inputs):
#         # 根据具体计算流程调用各个模块，这里仅为初始化示例
#         pass

# # 示例测试（假设某层输入符合对应尺寸）
# if __name__ == '__main__':
#     model = MyModel()
#     x_down_0_0 = torch.randn(4, 320, 64, 64)
#     # 示例调用其中一个融合模块
#     out = model.down_0_0(x_down_0_0, x_down_0_0)
#     print(out.shape)  # 应输出 torch.Size([4, 320, 64, 64])


import torch
x = torch.load('/data1/JM/code/BrushNet-main/exp/brushnet_adapter_/checkpoint-10/unet/diffusion_pytorch_model.bin')
print(x.keys())

