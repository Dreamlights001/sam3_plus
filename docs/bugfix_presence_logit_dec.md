# Bug Fix: presence_logit_dec 键缺失问题

## 问题描述
在使用 SAM3 模型进行异常检测时，程序在 `sam3_image_processor.py` 文件中尝试访问 `outputs["presence_logit_dec"]` 键时出现 `KeyError` 错误，导致程序无法继续执行。

## 错误信息
```
KeyError: 'presence_logit_dec'
```

错误发生在：`model/sam3/model/sam3_image_processor.py` 第 195 行

## 问题分析

1. 通过代码检查发现，`sam3_image_processor.py` 中的 `_forward_grounding` 方法尝试访问 `outputs["presence_logit_dec"]` 键。
2. 但实际查看 `sam3_image.py` 中的 `forward_grounding` 方法返回的 `outputs` 字典中，**并不包含** `presence_logit_dec` 键。
3. `outputs` 字典实际包含的键有：`pred_boxes`、`pred_logits`、`pred_masks` 等，但没有 `presence_logit_dec`。

## 解决方案

在 `sam3_image_processor.py` 文件中，添加对 `presence_logit_dec` 键存在性的检查。如果该键不存在，则使用默认值 `1.0` 生成 `presence_score`。

修改内容：
```python
# 在访问 outputs["presence_logit_dec"] 前添加检查
if "presence_logit_dec" in outputs:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
else:
    # 如果不存在，使用默认值 1.0
    presence_score = torch.ones_like(out_probs[..., :1])
```

## 测试验证

### 1. 极简测试脚本
创建了 `scripts/minimal_test.py` 脚本，直接测试修复后的逻辑：
- 创建模拟的 `outputs` 字典（不含 `presence_logit_dec`）
- 手动执行修复后的代码逻辑
- 验证代码能正常运行并生成正确形状的输出

### 2. 完整测试脚本
使用原始的测试脚本 `scripts/test.py` 进行测试，验证修复在完整场景中有效。

## 依赖安装

测试过程中发现缺少 `pycocotools` 依赖，已通过以下命令安装：
```bash
pip install pycocotools
```

## 结论

修复后，程序能够正确处理 `presence_logit_dec` 键缺失的情况，使用默认值 `1.0` 继续执行，避免了 `KeyError` 错误，测试脚本能够成功运行完成。

## 注意事项

- 此修复为临时解决方案，长期来看，应考虑在 `sam3_image.py` 中的 `forward_grounding` 方法中添加 `presence_logit_dec` 键的返回，以保持 API 的一致性。
- 建议后续更新文档，明确 `outputs` 字典中应该包含的所有键，以及每个键的含义。