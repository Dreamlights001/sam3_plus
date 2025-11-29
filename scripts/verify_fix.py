"""验证修复脚本：测试mask生成和坐标系显示修复"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("开始验证修复效果...")
    
    # 导入必要的模块
    import torch
    from model.sam3_anomaly_detector import SAM3AnomalyDetector
    
    # 创建输出目录
    output_dir = "./fix_verification"
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型路径
    model_path = r"D:\Users\Dlts\Documents\GitHub\sam3_plus\download\sam3"
    
    # 测试图像路径（使用示例图像，如果不存在则创建一个简单的测试图像）
    test_image_path = os.path.join(output_dir, "test_image.png")
    
    # 创建一个简单的测试图像
    if not os.path.exists(test_image_path):
        print(f"创建测试图像: {test_image_path}")
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # 白色背景
        # 添加一个红色方块作为异常
        test_image[200:300, 200:300, :] = [255, 0, 0]  # 红色方块
        Image.fromarray(test_image).save(test_image_path)
    
    # 初始化检测器
    print(f"初始化检测器: {model_path}")
    detector = SAM3AnomalyDetector(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 文本提示
    text_prompts = ["anomaly", "defect", "abnormality"]
    
    # 处理图像
    print(f"处理测试图像: {test_image_path}")
    image = detector.process_image(test_image_path)
    
    # 检测异常
    print("执行异常检测...")
    results = detector.detect_anomaly(image, text_prompts, conf_threshold=0.3)  # 使用较低的阈值以确保能检测到东西
    
    # 显示results的基本信息
    print("\n检测结果信息:")
    print(f"- masks数量: {len(results['masks'])}")
    print(f"- scores: {results['scores']}")
    
    # 获取异常mask
    print("\n生成异常mask...")
    anomaly_mask = detector.get_anomaly_mask(results, (512, 512))
    
    # 检查mask是否全黑
    non_zero_count = np.count_nonzero(anomaly_mask)
    print(f"\nMask统计信息:")
    print(f"- mask形状: {anomaly_mask.shape}")
    print(f"- 非零像素数: {non_zero_count}")
    print(f"- 像素值范围: [{anomaly_mask.min():.6f}, {anomaly_mask.max():.6f}]")
    print(f"- mask是否全黑: {non_zero_count == 0}")
    
    # 保存mask
    mask_path = os.path.join(output_dir, "anomaly_mask.png")
    # 归一化mask到0-255范围
    normalized_mask = (anomaly_mask - anomaly_mask.min()) / (anomaly_mask.max() - anomaly_mask.min() + 1e-8) * 255
    Image.fromarray(normalized_mask.astype(np.uint8)).save(mask_path)
    print(f"mask已保存到: {mask_path}")
    
    # 可视化结果
    print("\n生成可视化结果...")
    viz_path = os.path.join(output_dir, "visualization.png")
    detector.visualize_results(image, results, output_path=viz_path)
    print(f"可视化结果已保存到: {viz_path}")
    
    # 验证坐标系是否正确显示
    print("\n验证坐标系显示:")
    print("- 请手动检查visualization.png中的坐标系是否显示完整")
    print("- 预期结果：应该能看到x轴和y轴的刻度和标签")
    print("- 预期结果：图像应该正确显示，mask不应该全黑")
    
    print("\n验证完成！请检查以下文件：")
    print(f"1. {mask_path} - 检查是否不再是全黑图像")
    print(f"2. {viz_path} - 检查坐标系是否完整显示")
    
    # 显示修复评估结果
    if non_zero_count > 0:
        print("\n✅ 修复评估：mask全黑问题已解决！")
    else:
        print("\n❌ 修复评估：mask仍然是全黑的，请检查修复方案")
    
if __name__ == "__main__":
    main()