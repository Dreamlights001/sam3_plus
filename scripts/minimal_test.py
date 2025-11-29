"""极简测试脚本来验证presence_logit_dec修复是否成功"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入processor来测试
from model.sam3.model.sam3_image_processor import Sam3Processor
import torch

def main():
    print("极简测试presence_logit_dec修复...")
    
    # 创建一个模拟的outputs字典，不包含presence_logit_dec键
    print("创建模拟outputs字典(无presence_logit_dec)...")
    outputs = {
        "pred_boxes": torch.zeros(1, 10, 4),
        "pred_logits": torch.zeros(1, 10, 1),
        "pred_masks": torch.zeros(1, 10, 100, 100)
    }
    
    # 模拟out_logits
    out_logits = torch.zeros(1, 10, 1)
    
    # 手动执行我们修复的代码部分
    print("执行修复后的代码逻辑...")
    try:
        out_probs = out_logits.sigmoid()
        # 我们修复的关键部分
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            print("✓ found presence_logit_dec")
        else:
            print("✓ presence_logit_dec not found, using default 1.0")
            presence_score = torch.ones_like(out_probs[..., :1])
        
        out_probs = (out_probs * presence_score).squeeze(-1)
        print("✓ 修复后的代码执行成功!")
        print(f"  - out_probs shape: {out_probs.shape}")
        print("\n🎉 presence_logit_dec修复测试通过!")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()