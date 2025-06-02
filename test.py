import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw 
import numpy as np
import matplotlib.pyplot as plt

# 使用 from_pretrained 自动下载并加载模型
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
your_image = Image.open("1.jpeg")  
img_width, img_height = your_image.size

# 添加点提示 - 格式: [[x1, y1], [x2, y2], ...]
# 注意: 点的坐标应该在图像尺寸范围内
point_coords = np.array([
    [img_width // 2, img_height // 2],  # 图像中心点
])

# 添加点的标签 - 1表示前景点，0表示背景点
point_labels = np.array([1])  # 标记为前景点

# 添加边界框提示 - 格式: [x1, y1, x2, y2]
# 注意: 坐标为左上角和右下角的点
box = np.array([
    img_width // 4,          # 左上角x
    img_height // 4,         # 左上角y
    img_width * 3 // 4,      # 右下角x
    img_height * 3 // 4      # 右下角y
])

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(your_image)
    
    # 1. 仅使用点提示
    masks_points, iou_scores_points, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # 2. 仅使用框提示
    masks_box, iou_scores_box, _ = predictor.predict(
        box=box,
        multimask_output=True
    )
    
    # 3. 同时使用点和框提示
    masks_combined, iou_scores_combined, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True
    )

# 打印结果信息
print(f"点提示分割结果: {len(masks_points)} 个掩码, IoU: {iou_scores_points}")
print(f"框提示分割结果: {len(masks_box)} 个掩码, IoU: {iou_scores_box}")
print(f"组合提示分割结果: {len(masks_combined)} 个掩码, IoU: {iou_scores_combined}")

# 可视化提示和结果
def visualize_results(image, masks, iou_scores, points=None, point_labels=None, box=None, title=None):
    # 创建原始图像的副本
    result_img = image.copy().convert("RGBA")
    
    # 如果有点提示，绘制点
    if points is not None and point_labels is not None:
        point_img = Image.new("RGBA", result_img.size, (0, 0, 0, 0))
        point_draw = ImageDraw.Draw(point_img)
        
        for i, (coord, label) in enumerate(zip(points, point_labels)):
            x, y = int(coord[0]), int(coord[1])
            color = (0, 255, 0, 255) if label == 1 else (255, 0, 0, 255)  # 绿色为前景，红色为背景
            point_draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
        
        result_img = Image.alpha_composite(result_img, point_img)
    
    # 如果有框提示，绘制框
    if box is not None:
        box_img = Image.new("RGBA", result_img.size, (0, 0, 0, 0))
        box_draw = ImageDraw.Draw(box_img)
        box_draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=(255, 255, 0, 255), width=2)
        
        result_img = Image.alpha_composite(result_img, box_img)
    
    # 使用半透明蓝色绘制第一个掩码
    if len(masks) > 0:
        mask_img = Image.new("RGBA", result_img.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_img)
        
        mask_np = masks[0].astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize(result_img.size)
        
        for y in range(mask_pil.height):
            for x in range(mask_pil.width):
                if mask_pil.getpixel((x, y)) > 128:
                    mask_draw.point((x, y), fill=(0, 0, 255, 128))  # 半透明蓝色
        
        result_img = Image.alpha_composite(result_img, mask_img)
    
    result_img = result_img.convert("RGB")
    
    # 添加IoU分数文本
    if len(iou_scores) > 0:
        img_with_text = np.array(result_img)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_text)
        plt.title(title if title else "分割结果")
        plt.text(10, 30, f"IoU: {iou_scores[0]:.4f}", color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.5))
        plt.axis('off')
        
        # 保存并返回
        file_name = f"{title.replace(' ', '_')}.jpg"
        plt.savefig(file_name)
        print(f"结果已保存为 {file_name}")
        plt.close()
    
    return result_img

# 可视化三种不同提示的结果
visualize_results(your_image, masks_points, iou_scores_points, 
                  points=point_coords, point_labels=point_labels, 
                  title="点提示分割")

visualize_results(your_image, masks_box, iou_scores_box, 
                  box=box, 
                  title="框提示分割")

visualize_results(your_image, masks_combined, iou_scores_combined, 
                  points=point_coords, point_labels=point_labels, 
                  box=box, 
                  title="组合提示分割")

# 尝试显示图像窗口
try:
    # 在一个图中显示所有结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示点提示结果
    point_result = np.array(visualize_results(your_image, masks_points, iou_scores_points, 
                                              points=point_coords, point_labels=point_labels))
    axes[0].imshow(point_result)
    axes[0].set_title("点提示分割")
    axes[0].axis('off')
    
    # 显示框提示结果
    box_result = np.array(visualize_results(your_image, masks_box, iou_scores_box, box=box))
    axes[1].imshow(box_result)
    axes[1].set_title("框提示分割")
    axes[1].axis('off')
    
    # 显示组合提示结果
    combined_result = np.array(visualize_results(your_image, masks_combined, iou_scores_combined, 
                                                points=point_coords, point_labels=point_labels, box=box))
    axes[2].imshow(combined_result)
    axes[2].set_title("组合提示分割")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("all_prompt_results.jpg")
    print("所有提示分割结果已保存为 all_prompt_results.jpg")
    plt.show()
except Exception as e:
    print(f"无法显示图像窗口: {e}，但图像已保存到文件")