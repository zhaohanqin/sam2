import sys
import os
import numpy as np
import torch
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup,
                            QSlider, QGroupBox, QMessageBox, QSplitter, QProgressDialog, QProgressBar,
                            QStyle, QDialog, QFormLayout, QComboBox, QCheckBox, QDialogButtonBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QRect, QPoint
from PIL import Image, ImageDraw
from sam2.sam2_image_predictor import SAM2ImagePredictor
import traceback
import requests
import time
import platform
import urllib3

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量
COLORS = {
    "background": "#F5F7FA",        # 背景色，更浅的灰白色
    "card_bg": "#FFFFFF",           # 卡片背景色，纯白色
    "primary": "#4A6FDE",           # 主色调，更柔和的蓝色
    "primary_light": "#7B97E8",     # 主色调亮色
    "primary_dark": "#2A4BAE",      # 主色调暗色
    "secondary": "#26A69A",         # 辅助色，薄荷绿
    "secondary_light": "#64D8CB",   # 辅助色亮色
    "accent": "#FFB74D",            # 强调色，柔和的橙色
    "text_primary": "#2D3748",      # 主要文本色，深灰近黑
    "text_secondary": "#718096",    # 次要文本色，中灰色
    "divider": "#E2E8F0",           # 分隔线颜色，浅灰色
    "success": "#68D391",           # 成功色，柔和的绿色
    "warning": "#F6AD55",           # 警告色，柔和的橙色
    "error": "#FC8181",             # 错误色，柔和的红色
    "shadow": "rgba(0, 0, 0, 0.1)"  # 阴影颜色，透明黑
}

# 设置全局字体样式
FONT_FAMILY = "微软雅黑"      # 主字体
FONT_SIZE = 13                # 基础字体大小 (增大)
TITLE_FONT_SIZE = 18          # 标题字体大小 (增大)
BUTTON_FONT_SIZE = 14         # 按钮字体大小 (增大)
GROUP_TITLE_FONT_SIZE = 16    # 选项组标题字体大小 (增大)
RADIO_BUTTON_FONT_SIZE = 13   # 单选按钮字体大小 (增大)

# 定义模型信息
MODEL_INFO = {
    "tiny": {
        "name": "sam2.1-hiera-tiny",
        "repo": "facebook/sam2.1-hiera-tiny",
        "file_name": "sam2.1_hiera_tiny.pt",
        "download_url": "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
        "description": "最小 (适合低配置设备)"
    },
    "small": {
        "name": "sam2.1-hiera-small",
        "repo": "facebook/sam2.1-hiera-small",
        "file_name": "sam2.1_hiera_small.pt",
        "download_url": "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt",
        "description": "小型 (适合中等配置)"
    },
    "base": {
        "name": "sam2.1-hiera-base",
        "repo": "facebook/sam2.1-hiera-base",
        "file_name": "sam2.1_hiera_base.pt",
        "download_url": "https://huggingface.co/facebook/sam2.1-hiera-base/resolve/main/sam2.1_hiera_base.pt",
        "description": "中型 (推荐配置)"
    },
    "large": {
        "name": "sam2.1-hiera-large",
        "repo": "facebook/sam2.1-hiera-large",
        "file_name": "sam2.1_hiera_large.pt",
        "download_url": "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt",
        "description": "大型 (高端配置)"
    }
}

# 获取模型存储路径
def get_model_dir():
    """获取模型存储目录，使用当前程序所在目录下的models文件夹"""
    # 获取程序所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录
    project_root = os.path.dirname(script_dir)
    
    # 在项目根目录下创建models文件夹
    model_dir = os.path.join(project_root, "models")
    
    # 如果目录不存在，创建它
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Using models directory: {model_dir}")
    return model_dir

# PIL图像转换为QImage的辅助函数
def pil_to_qimage(pil_image):
    """将PIL图像转换为QImage，使用更安全的方法"""
    try:
        # 确保图像是RGB或RGBA模式
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")
            
        # 获取图像尺寸
        width, height = pil_image.size
        
        # 检查图像尺寸是否过大，如果过大则进一步缩小
        MAX_QT_DIMENSION = 8000  # Qt的图像尺寸限制
        if width > MAX_QT_DIMENSION or height > MAX_QT_DIMENSION:
            scale = min(MAX_QT_DIMENSION / width, MAX_QT_DIMENSION / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            logger.warning(f"Image too large for Qt ({width}x{height}), resizing to {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            width, height = pil_image.size
        
        if pil_image.mode == "RGBA":
            # 对于RGBA模式
            fmt = QImage.Format_RGBA8888
            bytes_per_line = 4 * width
            buffer = pil_image.tobytes("raw", "RGBA")
        else:
            # 对于RGB模式
            fmt = QImage.Format_RGB888
            bytes_per_line = 3 * width
            buffer = pil_image.tobytes("raw", "RGB")
        
        # 创建QImage并复制数据
        result = QImage(buffer, width, height, bytes_per_line, fmt).copy()
        return result
    except Exception as e:
        logger.error(f"Error in pil_to_qimage: {str(e)}", exc_info=True)
        # 返回一个小的空白图像，防止程序崩溃
        return QImage(10, 10, QImage.Format_RGB888)

class ImageCanvas(QLabel):
    """用于显示图像和处理交互的画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(600, 400)  # 减小最小高度
        self.setMaximumHeight(450)     # 添加最大高度限制
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 8px;")
        
        # 图像相关
        self.image = None
        self.display_image = None
        self.scaled_image = None
        self.result_image = None
        
        # 交互状态
        self.drawing = False
        self.point_mode = True  # True: 点模式, False: 框模式
        self.foreground_point = True  # True: 前景点, False: 背景点
        
        # 点和框数据
        self.points = []  # 格式: [(x, y, is_foreground), ...]
        self.point_labels = []  # 1: 前景, 0: 背景
        self.box = None  # 格式: [x1, y1, x2, y2]
        self.current_box = None  # 当前正在绘制的框
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        logger.info("ImageCanvas initialized")
    
    def set_point_mode(self, is_point_mode):
        """设置是否为点模式"""
        self.point_mode = is_point_mode
        logger.info(f"Point mode set to: {is_point_mode}")
    
    def set_foreground_point(self, is_foreground):
        """设置是否为前景点"""
        self.foreground_point = is_foreground
        logger.info(f"Foreground point set to: {is_foreground}")
    
    def clear_prompts(self):
        """清除所有提示"""
        self.points = []
        self.point_labels = []
        self.box = None
        self.current_box = None
        self.update_display()
        logger.info("All prompts cleared")
    
    def load_image(self, image_path):
        """加载图像，使用更安全的方法"""
        try:
            logger.info(f"Loading image from: {image_path}")
            
            # 先获取图像信息，不完全加载到内存
            with Image.open(image_path) as img:
                width, height = img.size
                logger.info(f"Original image size: {width}x{height}")
                
                # 检查图像尺寸是否过大 - 进一步降低最大尺寸限制
                MAX_DIMENSION = 1024  # 降低最大尺寸到1024
                MAX_PIXELS = 1000000  # 降低最大像素数到100万像素
                
                # 计算图像总像素数
                total_pixels = width * height
                logger.info(f"Total pixels: {total_pixels}")
                
                # 如果图像尺寸过大，进行缩放
                if total_pixels > MAX_PIXELS or width > MAX_DIMENSION or height > MAX_DIMENSION:
                    # 计算缩放比例
                    scale_factor = min(
                        1.0,
                        MAX_DIMENSION / max(width, height),
                        (MAX_PIXELS / total_pixels) ** 0.5
                    )
                    
                    # 计算新尺寸
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    logger.info(f"Scale factor: {scale_factor}, new size: {new_width}x{new_height}")
                    
                    # 提示用户图像将被缩放
                    QMessageBox.information(
                        self, 
                        "图像缩放", 
                        f"图像尺寸过大 ({width}x{height})，将自动缩放至 ({new_width}x{new_height}) 以避免系统崩溃。"
                    )
                    
                    # 处理Qt事件，确保UI响应
                    QApplication.processEvents()
                    
                    try:
                        # 使用分块处理来减少内存使用
                        logger.info(f"Resizing image to: {new_width}x{new_height}")
                        
                        # 使用NEAREST替代BILINEAR，最小内存占用
                        self.image = img.resize((new_width, new_height), Image.Resampling.NEAREST)
                        
                        # 确保转换为RGB模式
                        if self.image.mode != "RGB":
                            self.image = self.image.convert("RGB")
                            
                        logger.info(f"Image resized successfully")
                    except Exception as resize_error:
                        logger.error(f"Error during image resize: {str(resize_error)}")
                        # 最后的尝试：创建一个新的空白图像
                        try:
                            logger.info("Creating new empty image")
                            self.image = Image.new("RGB", (new_width, new_height), (255, 255, 255))
                            logger.info("Empty image created successfully")
                        except Exception as create_error:
                            logger.error(f"Create empty image failed: {str(create_error)}")
                            raise
                else:
                    # 图像尺寸合适，直接加载
                    logger.info("Image size is acceptable, loading directly")
                    self.image = img.copy().convert("RGB")
                    logger.info("Image loaded successfully")
            
            # 强制进行垃圾回收
            import gc
            gc.collect()
            
            # 使用QApplication.processEvents()确保UI响应
            QApplication.processEvents()
            
            logger.info("Updating display with loaded image")
            self.update_display()
            logger.info("Display updated successfully")
            
            # 再次强制垃圾回收
            gc.collect()
            
            return True
        except MemoryError as me:
            logger.error(f"Memory error loading image: {str(me)}", exc_info=True)
            QMessageBox.critical(self, "内存错误", "图像太大，内存不足。请尝试使用更小的图像。")
            return False
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}", exc_info=True)
            # 打印完整的异常堆栈
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            return False
    
    def update_display(self):
        """更新显示的图像，使用更安全的方法"""
        if self.image is None:
            logger.warning("Cannot update display, image is None")
            return
        
        try:
            # 创建工作副本，但限制大小
            MAX_DISPLAY_DIMENSION = 1920
            img_width, img_height = self.image.size
            
            # 如果图像太大，先缩小再处理
            if img_width > MAX_DISPLAY_DIMENSION or img_height > MAX_DISPLAY_DIMENSION:
                scale = min(MAX_DISPLAY_DIMENSION / img_width, MAX_DISPLAY_DIMENSION / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                logger.info(f"Scaling display image to {new_width}x{new_height}")
                
                # 使用BILINEAR方法缩放，速度和内存占用的平衡
                display_img = self.image.resize((new_width, new_height), Image.Resampling.BILINEAR)
                self.display_image = display_img
            else:
                # 图像尺寸合适，直接复制
                self.display_image = self.image.copy()
            
            # 绘制点
            if self.points:
                for x, y, is_foreground in self.points:
                    color = (0, 255, 0) if is_foreground else (255, 0, 0)  # 绿色为前景，红色为背景
                    self._draw_point(self.display_image, x, y, color)
            
            # 绘制框
            if self.box:
                self._draw_box(self.display_image, self.box)
            
            # 绘制当前正在绘制的框
            if self.current_box:
                self._draw_box(self.display_image, self.current_box)
            
            # 转换为QPixmap并显示
            self._update_pixmap()
            
            # 强制垃圾回收
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Error in update_display: {str(e)}", exc_info=True)
            QMessageBox.warning(self.parent, "警告", f"更新显示时出错: {str(e)}")
    
    def _draw_point(self, img, x, y, color, radius=8):  # 增大点的半径
        """在图像上绘制点"""
        draw = ImageDraw.Draw(img)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    
    def _draw_box(self, img, box):
        """在图像上绘制框"""
        # 确保坐标正确排序：左上角坐标小于右下角坐标
        x0, y0, x1, y1 = box
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)  # 增加线宽
    
    def _update_pixmap(self):
        """更新QPixmap，使用更安全的方法"""
        if self.display_image is None:
            logger.warning("Cannot update pixmap, display_image is None")
            return
        
        try:
            # 转换PIL图像为QImage
            logger.debug("Converting PIL image to QImage")
            qim = pil_to_qimage(self.display_image)
            
            # 检查QImage是否有效
            if qim.isNull():
                logger.error("Generated QImage is null")
                return
            
            # 根据控件大小缩放图像
            logger.debug("Scaling image to fit canvas")
            max_width = min(self.width(), 2000)  # 限制最大宽度
            max_height = min(self.height(), 2000)  # 限制最大高度
            
            # 使用更安全的缩放方法
            self.scaled_image = qim.scaled(
                max_width, 
                max_height, 
                Qt.KeepAspectRatio, 
                Qt.FastTransformation  # 使用更快的转换方法
            )
            
            # 创建QPixmap并设置
            logger.debug("Creating QPixmap from QImage")
            pixmap = QPixmap.fromImage(self.scaled_image)
            
            # 检查pixmap是否有效
            if pixmap.isNull():
                logger.error("Generated QPixmap is null")
                return
            
            self.setPixmap(pixmap)
            logger.debug("Pixmap set successfully")
            
            # 处理Qt事件，确保UI响应
            QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating pixmap: {str(e)}", exc_info=True)
            # 显示错误信息，但不中断程序
            if self.parent:
                QMessageBox.warning(self.parent, "警告", f"图像显示出错: {str(e)}")
    
    def get_image_coordinates(self, pos):
        """将控件坐标转换为图像坐标"""
        if self.image is None or self.scaled_image is None:
            return None
        
        # 获取图像在控件中的位置
        img_rect = QRect(
            (self.width() - self.scaled_image.width()) // 2,
            (self.height() - self.scaled_image.height()) // 2,
            self.scaled_image.width(),
            self.scaled_image.height()
        )
        
        # 检查点击是否在图像内
        if not img_rect.contains(pos):
            return None
        
        # 计算相对于图像的坐标
        x_ratio = self.image.width / self.scaled_image.width()
        y_ratio = self.image.height / self.scaled_image.height()
        
        x = int((pos.x() - img_rect.left()) * x_ratio)
        y = int((pos.y() - img_rect.top()) * y_ratio)
        
        return (x, y)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image is None:
            return
        
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
        
        if self.point_mode:
            # 点模式：添加点
            self.points.append((pos[0], pos[1], self.foreground_point))
            self.point_labels.append(1 if self.foreground_point else 0)
            logger.info(f"Added point at {pos}, foreground: {self.foreground_point}")
            self.update_display()
        else:
            # 框模式：开始绘制框
            self.drawing = True
            self.current_box = [pos[0], pos[1], pos[0], pos[1]]
            logger.info(f"Started drawing box at {pos}")
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.image is None or not self.drawing or self.point_mode:
            return
        
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
        
        # 更新当前框的终点
        self.current_box[2] = pos[0]
        self.current_box[3] = pos[1]
        self.update_display()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.image is None or not self.drawing or self.point_mode:
            return
        
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
        
        # 完成框的绘制
        self.drawing = False
        self.box = [
            min(self.current_box[0], pos[0]),
            min(self.current_box[1], pos[1]),
            max(self.current_box[0], pos[0]),
            max(self.current_box[1], pos[1])
        ]
        logger.info(f"Finished drawing box: {self.box}")
        self.current_box = None
        self.update_display()
    
    def resizeEvent(self, event):
        """控件大小改变事件"""
        super().resizeEvent(event)
        self._update_pixmap()
    
    def get_prompt_data(self):
        """获取提示数据用于分割"""
        point_coords = None
        point_labels = None
        box = None
        
        if self.points:
            points = [(x, y) for x, y, _ in self.points]
            point_coords = np.array(points)
            point_labels = np.array(self.point_labels)
            logger.info(f"Prompt data - points: {len(points)}, labels: {point_labels}")
        
        if self.box:
            box = np.array(self.box)
            logger.info(f"Prompt data - box: {box}")
        
        return point_coords, point_labels, box
    
    def set_result_image(self, result_image):
        """设置并显示结果图像"""
        self.result_image = result_image
        if result_image is None:
            return
        qim = pil_to_qimage(result_image)
        self.scaled_image = qim.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.scaled_image))


class ResultCanvas(QLabel):
    """用于显示分割结果的画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent  # 保存父对象引用
        self.setMinimumSize(600, 400)  # 减小最小高度
        self.setMaximumHeight(450)     # 添加最大高度限制
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 8px;")
        self.result_image = None
        self.scaled_image = None
        logger.info("ResultCanvas initialized")
    
    def set_result(self, result_image):
        """设置结果图像"""
        self.result_image = result_image
        if result_image is None:
            logger.info("Result image cleared")
            self.clear()
        else:
            logger.info("Setting result image")
            self._update_pixmap()
    
    def _update_pixmap(self):
        """更新QPixmap，使用更安全的方法"""
        if self.result_image is None:
            return
        
        try:
            # 转换PIL图像为QImage
            logger.debug("Converting result image to QImage")
            qim = pil_to_qimage(self.result_image)
            
            # 检查QImage是否有效
            if qim.isNull():
                logger.error("Generated result QImage is null")
                return
            
            # 根据控件大小缩放图像
            logger.debug("Scaling result image")
            max_width = min(self.width(), 2000)  # 限制最大宽度
            max_height = min(self.height(), 2000)  # 限制最大高度
            
            # 使用更安全的缩放方法
            self.scaled_image = qim.scaled(
                max_width, 
                max_height, 
                Qt.KeepAspectRatio, 
                Qt.FastTransformation  # 使用更快的转换方法
            )
            
            logger.debug("Creating QPixmap from scaled image")
            pixmap = QPixmap.fromImage(self.scaled_image)
            
            # 检查pixmap是否有效
            if pixmap.isNull():
                logger.error("Generated result QPixmap is null")
                return
            
            self.setPixmap(pixmap)
            logger.debug("Result pixmap set successfully")
            
            # 处理Qt事件，确保UI响应
            QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating result pixmap: {str(e)}", exc_info=True)
            # 显示错误信息，但不中断程序
            if self.parent:
                QMessageBox.warning(self.parent, "警告", f"结果图像显示出错: {str(e)}")
    
    def resizeEvent(self, event):
        """控件大小改变事件"""
        super().resizeEvent(event)
        self._update_pixmap()


class SAM2UI(QMainWindow):
    """SAM2交互式分割界面"""
    def __init__(self):
        try:
            super().__init__()
            self.predictor = None
            self.current_mask = None  # 存储当前的掩码
            self.current_obj_id = 1   # 当前对象ID
            logger.info("Initializing SAM2UI")
            self.initUI()
        except Exception as e:
            logger.critical(f"Failed to initialize UI: {str(e)}", exc_info=True)
            QMessageBox.critical(None, "严重错误", f"初始化界面失败: {str(e)}")
            sys.exit(1)
    
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('SAM2 图像智能分割')
        self.setGeometry(100, 100, 1600, 950)  # 增加窗口尺寸
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QGroupBox {{
                border: 1px solid {COLORS['divider']};
                border-radius: 10px;
                margin-top: 30px;  /* 增加顶部边距，确保标题完全显示 */
                font-weight: 500;
                background-color: {COLORS['card_bg']};
                padding: 15px;  /* 增加内边距 */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                top: -15px;  /* 调整位置 */
                padding: 0 10px;
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_primary']};
                font-size: {GROUP_TITLE_FONT_SIZE}px;
                font-weight: bold;
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: {FONT_SIZE}px;  /* 使用更大的基础字体 */
            }}
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {COLORS['divider']};
                text-align: center;
                height: 10px;  /* 增加高度 */
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['divider']};
                height: 10px;  /* 增加高度 */
                background: {COLORS['divider']};
                margin: 2px 0;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['primary']};
                border: 1px solid {COLORS['primary']};
                width: 22px;  /* 增加宽度 */
                height: 22px;  /* 增加高度 */
                margin: -7px 0;
                border-radius: 11px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS['primary_light']};
            }}
        """)
        
        # 设置应用字体
        app_font = QFont(FONT_FAMILY, FONT_SIZE)
        app_font.setStyleStrategy(QFont.PreferAntialias)  # 使字体更平滑
        QApplication.setFont(app_font)
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)  # 增加布局间距
        main_layout.setContentsMargins(30, 30, 30, 30)  # 增加边距
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 创建顶部导航栏
        self.create_top_navigation(main_layout)
        
        # 创建内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)  # 增加间距
        
        # 创建左侧工具面板
        self.create_tool_panel(content_layout)
        
        # 创建右侧画布区域
        self.create_canvas_area(content_layout)
        
        # 添加内容区域到主布局
        main_layout.addLayout(content_layout, 1)  # 1是拉伸因子，让内容区域占据更多空间
        
        # 创建状态栏
        self.statusBar().showMessage('就绪')
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['divider']};
                padding: 8px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE}px;
                min-height: 24px;
            }}
        """)
        
        # 添加进度条到状态栏
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        logger.info("UI initialization complete")
    
    def create_top_navigation(self, parent_layout):
        """创建顶部导航栏"""
        nav_bar = QWidget()
        nav_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['card_bg']};
                border-radius: 10px;
                border: 1px solid {COLORS['divider']};
            }}
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(20, 15, 20, 15)
        nav_layout.setSpacing(25)
        
        # 创建标题标签
        title_label = QLabel("SAM2 图像智能分割")
        title_label.setStyleSheet(f"""
            font-size: {TITLE_FONT_SIZE}px;
            font-weight: bold;
            color: {COLORS['primary']};
            padding: 5px 10px;
        """)
        nav_layout.addWidget(title_label)
        
        # 添加弹性空间
        nav_layout.addStretch(1)
        
        # 统一按钮样式
        button_style = f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 8px;  /* 增大圆角 */
                padding: 12px 18px;  /* 增加内边距 */
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: 500;
                min-width: 130px;  /* 增加最小宽度 */
                min-height: 42px;  /* 增加最小高度 */
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['divider']};
                color: {COLORS['text_secondary']};
            }}
        """
        
        # 加载图像按钮
        load_btn = QPushButton("加载图像")
        load_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogOpenButton")))
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet(button_style)
        nav_layout.addWidget(load_btn)
        
        # 批量处理按钮
        batch_btn = QPushButton("批量处理")
        batch_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DirIcon")))
        batch_btn.clicked.connect(self.batch_process)
        batch_btn.setStyleSheet(button_style)
        nav_layout.addWidget(batch_btn)
        
        # 加载模型按钮
        load_model_btn = QPushButton("加载模型")
        load_model_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogApplyButton")))
        load_model_btn.clicked.connect(self.load_model)
        load_model_btn.setStyleSheet(button_style)
        nav_layout.addWidget(load_model_btn)
        
        # 保存结果按钮
        save_btn = QPushButton("保存结果")
        save_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogSaveButton")))
        save_btn.clicked.connect(self.save_result)
        save_btn.setStyleSheet(button_style)
        nav_layout.addWidget(save_btn)
        
        parent_layout.addWidget(nav_bar)
    
    def create_tool_panel(self, parent_layout):
        """创建左侧工具面板"""
        tool_panel = QWidget()
        tool_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['card_bg']};
                border-radius: 10px;
                border: 1px solid {COLORS['divider']};
            }}
        """)
        tool_panel.setMaximumWidth(350)  # 增加最大宽度
        tool_panel.setMinimumWidth(330)  # 增加最小宽度
        
        tool_layout = QVBoxLayout(tool_panel)
        tool_layout.setContentsMargins(25, 25, 25, 25)  # 增加边距
        tool_layout.setSpacing(30)  # 增加间距
        
        # 创建分割工具组 - 增大标题和内容间距，但去掉标题文字
        segmentation_group = QGroupBox()  # 移除标题文字
        segmentation_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {COLORS['divider']};
                border-radius: 10px;
                margin-top: 10px;  /* 减小顶部边距，因为没有标题了 */
                padding-top: 10px;
                background-color: {COLORS['card_bg']};
            }}
        """)
        
        segmentation_layout = QVBoxLayout()
        segmentation_layout.setSpacing(25)  # 增加间距
        segmentation_layout.setContentsMargins(20, 30, 20, 20)  # 增加内边距
        
        # 提示类型选择 - 增大字体
        prompt_type_label = QLabel("提示类型:")
        prompt_type_label.setStyleSheet(f"""
            font-weight: 600; 
            color: {COLORS['text_primary']};
            font-size: 16px;  /* 增大字体 */
            margin-top: 10px;
        """)
        segmentation_layout.addWidget(prompt_type_label)
        
        # 提示类型单选按钮 - 修改布局和样式，确保文字完全显示
        prompt_type_container = QWidget()
        prompt_type_layout = QVBoxLayout(prompt_type_container)
        prompt_type_layout.setContentsMargins(5, 0, 5, 0)
        prompt_type_layout.setSpacing(15)  # 增加垂直间距
        
        # 更新单选按钮样式，确保宽度足够
        radio_style = f"""
            QRadioButton {{
                font-size: {RADIO_BUTTON_FONT_SIZE + 2}px;  /* 增大字体 */
                color: {COLORS['text_primary']};
                spacing: 12px;  /* 增加文字与按钮的间距 */
                padding: 10px;  /* 增加内边距 */
                min-width: 120px;  /* 确保宽度足够显示文字 */
            }}
            QRadioButton::indicator {{
                width: 22px;  /* 增大按钮尺寸 */
                height: 22px;
                border-radius: 11px;
                border: 2px solid {COLORS['primary']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {COLORS['primary']};
                border: 2px solid {COLORS['primary']};
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWxpbmUgcG9pbnRzPSIyMCA2IDkgMTcgNCAxMiI+PC9wb2x5bGluZT48L3N2Zz4=);
                background-position: center;
                background-repeat: no-repeat;
            }}
            QRadioButton:hover {{
                background-color: {COLORS['divider']};
                border-radius: 6px;
            }}
        """
        
        # 为每个单选按钮使用单独的行，确保完整显示
        # 点标注按钮
        point_radio_container = QWidget()
        point_radio_layout = QHBoxLayout(point_radio_container)
        point_radio_layout.setContentsMargins(0, 0, 0, 0)
        
        self.point_radio = QRadioButton("点标注")
        self.point_radio.setStyleSheet(radio_style)
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(self.toggle_prompt_type)
        
        point_radio_layout.addWidget(self.point_radio)
        point_radio_layout.addStretch(1)
        prompt_type_layout.addWidget(point_radio_container)
        
        # 框标注按钮
        box_radio_container = QWidget()
        box_radio_layout = QHBoxLayout(box_radio_container)
        box_radio_layout.setContentsMargins(0, 0, 0, 0)
        
        self.box_radio = QRadioButton("框标注")
        self.box_radio.setStyleSheet(radio_style)
        self.box_radio.toggled.connect(self.toggle_prompt_type)
        
        box_radio_layout.addWidget(self.box_radio)
        box_radio_layout.addStretch(1)
        prompt_type_layout.addWidget(box_radio_container)
        
        segmentation_layout.addWidget(prompt_type_container)
        
        # 第二行 - 语义分割
        prompt_type_row2 = QHBoxLayout()
        prompt_type_row2.setSpacing(15)
        
        self.semantic_radio = QRadioButton("语义分割")
        self.semantic_radio.setStyleSheet(radio_style)
        self.semantic_radio.toggled.connect(self.toggle_prompt_type)
        
        prompt_type_row2.addWidget(self.semantic_radio)
        prompt_type_row2.addStretch(1)
        
        prompt_type_layout.addLayout(prompt_type_row2)
        
        # Line commented out to fix error: segmentation_layout.addWidget(self.semantic_options_container)
        
        # 点类型选择（仅在点模式下显示）- 增大字体
        self.point_type_container = QWidget()
        point_type_layout = QVBoxLayout(self.point_type_container)
        point_type_layout.setContentsMargins(0, 5, 0, 5)  # 调整边距
        point_type_layout.setSpacing(15)  # 增加间距
        
        point_type_label = QLabel("点类型:")
        point_type_label.setStyleSheet(f"""
            font-weight: 600; 
            color: {COLORS['text_primary']};
            font-size: 16px;  /* 增大字体 */
            margin-top: 5px;
        """)
        point_type_layout.addWidget(point_type_label)
        
        # 点类型单选按钮
        point_type_radio_container = QWidget()
        point_type_radio_layout = QHBoxLayout(point_type_radio_container)
        point_type_radio_layout.setContentsMargins(5, 0, 5, 0)
        point_type_radio_layout.setSpacing(20)
        
        self.foreground_radio = QRadioButton("前景")
        self.foreground_radio.setStyleSheet(radio_style)
        self.foreground_radio.setChecked(True)
        self.foreground_radio.toggled.connect(self.toggle_point_type)
        
        self.background_radio = QRadioButton("背景")
        self.background_radio.setStyleSheet(radio_style)
        
        point_type_radio_layout.addWidget(self.foreground_radio)
        point_type_radio_layout.addWidget(self.background_radio)
        point_type_radio_layout.addStretch(1)
        point_type_layout.addWidget(point_type_radio_container)
        
        segmentation_layout.addWidget(self.point_type_container)
        
        # 语义分割选项（仅在语义分割模式下显示）
        self.semantic_options_container = QWidget()
        self.semantic_options_container.setVisible(False)  # 默认隐藏
        semantic_options_layout = QVBoxLayout(self.semantic_options_container)
        semantic_options_layout.setContentsMargins(0, 0, 0, 0)
        semantic_options_layout.setSpacing(15)
        
        # 语义分割说明
        semantic_desc = QLabel("自动对整个图像进行语义分割，识别所有物体")
        semantic_desc.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 5px;
            background-color: {COLORS['divider']};
            border-radius: 4px;
            line-height: 1.4;
        """)
        semantic_desc.setWordWrap(True)
        semantic_options_layout.addWidget(semantic_desc)
        
        # 分割阈值
        threshold_label = QLabel("分割阈值:")
        threshold_label.setStyleSheet(f"""
            font-weight: 500; 
            color: {COLORS['text_primary']};
            font-size: 13px;
        """)
        semantic_options_layout.addWidget(threshold_label)
        
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(10)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)  # 默认值50%
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        
        self.threshold_value_label = QLabel("50%")
        self.threshold_value_label.setStyleSheet(f"""
            color: {COLORS['primary']};
            font-weight: bold;
            min-width: 40px;
            text-align: right;
        """)
        
        # 连接滑块值变化信号
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        
        semantic_options_layout.addWidget(threshold_container)
        
        segmentation_layout.addWidget(self.semantic_options_container)
        
        # 添加分隔线
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {COLORS['divider']};")
        segmentation_layout.addWidget(separator)
        
        # 对象ID控制 - 增大字体
        obj_id_label = QLabel("对象ID:")
        obj_id_label.setStyleSheet(f"""
            font-weight: 600; 
            color: {COLORS['text_primary']};
            font-size: 16px;  /* 增大字体 */
            margin-top: 10px;
        """)
        segmentation_layout.addWidget(obj_id_label)
        
        obj_id_container = QWidget()
        obj_id_layout = QHBoxLayout(obj_id_container)
        obj_id_layout.setContentsMargins(5, 5, 5, 5)
        obj_id_layout.setSpacing(20)  # 增加间距
        
        # 对象ID显示 - 增大字体和尺寸
        self.obj_id_label = QLabel(f"{self.current_obj_id}")
        self.obj_id_label.setStyleSheet(f"""
            font-size: 22px;  /* 增大字体 */
            font-weight: bold;
            color: {COLORS['primary']};
            background-color: {COLORS['divider']};
            border-radius: 6px;
            padding: 8px 25px;
            min-width: 50px;
            text-align: center;
        """)
        self.obj_id_label.setAlignment(Qt.AlignCenter)
        
        # ID控制按钮样式
        id_button_style = f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['divider']};
                color: {COLORS['text_secondary']};
            }}
        """
        
        # 减少ID按钮
        decrease_id_btn = QPushButton("-")
        decrease_id_btn.clicked.connect(self.decrease_obj_id)
        decrease_id_btn.setStyleSheet(id_button_style)
        
        # 增加ID按钮
        increase_id_btn = QPushButton("+")
        increase_id_btn.clicked.connect(self.increase_obj_id)
        increase_id_btn.setStyleSheet(id_button_style)
        
        obj_id_layout.addStretch(1)
        obj_id_layout.addWidget(decrease_id_btn)
        obj_id_layout.addWidget(self.obj_id_label)
        obj_id_layout.addWidget(increase_id_btn)
        obj_id_layout.addStretch(1)
        
        segmentation_layout.addWidget(obj_id_container)
        
        # 添加分隔线
        separator2 = QWidget()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet(f"background-color: {COLORS['divider']};")
        segmentation_layout.addWidget(separator2)
        
        # 操作按钮
        action_button_style = f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 8px;  /* 增大圆角 */
                padding: 14px;  /* 增加内边距 */
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: 500;
                min-height: 48px;  /* 增加最小高度 */
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_light']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['secondary']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['divider']};
                color: {COLORS['text_secondary']};
            }}
        """
        
        action_buttons_layout = QVBoxLayout()
        action_buttons_layout.setSpacing(15)
        
        # 分割按钮
        segment_btn = QPushButton("分割")
        segment_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_MediaPlay")))
        segment_btn.clicked.connect(self.perform_segmentation)
        segment_btn.setStyleSheet(action_button_style)
        action_buttons_layout.addWidget(segment_btn)
        
        # 清除提示按钮
        clear_btn = QPushButton("清除提示")
        clear_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_DialogResetButton")))
        clear_btn.clicked.connect(self.clear_prompts)
        clear_btn.setStyleSheet(action_button_style)
        action_buttons_layout.addWidget(clear_btn)
        
        # 切换视图按钮
        toggle_view_btn = QPushButton("切换视图")
        toggle_view_btn.setIcon(self.style().standardIcon(getattr(QStyle, "SP_ComputerIcon")))
        toggle_view_btn.clicked.connect(self.toggle_result_view)
        toggle_view_btn.setStyleSheet(action_button_style)
        action_buttons_layout.addWidget(toggle_view_btn)
        
        segmentation_layout.addLayout(action_buttons_layout)
        segmentation_layout.addStretch(1)
        
        segmentation_group.setLayout(segmentation_layout)
        tool_layout.addWidget(segmentation_group)
        
        # 添加帮助提示
        help_label = QLabel("提示：先加载图像和模型，然后添加点或框标记目标区域，最后点击分割按钮")
        help_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 12px;
            background-color: {COLORS['divider']};
            border-radius: 5px;
            line-height: 1.4;
        """)
        help_label.setWordWrap(True)
        tool_layout.addWidget(help_label)
        
        # 添加到父布局
        parent_layout.addWidget(tool_panel)
    
    def create_canvas_area(self, parent_layout):
        """创建右侧画布区域"""
        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['card_bg']};
                border-radius: 10px;
                border: 1px solid {COLORS['divider']};
            }}
        """)
        
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(2, 1, 20, 20)  # 减小顶部边距
        canvas_layout.setSpacing(1)  # 减小间距
        
        # 创建画布标题 - 居中显示但减小边距
        canvas_title = QLabel("图像编辑区域")
        canvas_title.setAlignment(Qt.AlignCenter)  # 居中对齐
        canvas_title.setStyleSheet(f"""
            font-size: 40px;  /* 保持大字体 */
            font-weight: bold;
            color: {COLORS['primary']};
            padding-bottom: 40px;  /* 减小底部边距 */
            padding-top: 2px;  /* 减小顶部边距 */
            border-bottom: 1px solid {COLORS['divider']};
            text-align: center;
        """)
        canvas_layout.addWidget(canvas_title)
        
        # 创建分割器，允许调整两个画布的大小
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)  # 增加分割条宽度
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['divider']};
                border-radius: 4px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['primary_light']};
            }}
        """)
        
        # 原始图像画布容器
        original_container = QWidget()
        original_container.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border-radius: 8px;
            border: 1px solid {COLORS['divider']};
        """)
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(10, 10, 10, 10)
        
        # 原始图像标签
        original_label = QLabel("原始图像")
        original_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 500;
            color: {COLORS['text_primary']};
            padding: 5px;
            background-color: {COLORS['card_bg']};
            border-radius: 4px;
        """)
        original_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_label)
        
        # 原始图像画布
        self.image_canvas = ImageCanvas()
        self.image_canvas.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border-radius: 5px;
        """)
        original_layout.addWidget(self.image_canvas)
        
        # 结果图像画布容器
        result_container = QWidget()
        result_container.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border-radius: 8px;
            border: 1px solid {COLORS['divider']};
        """)
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(10, 10, 10, 10)
        
        # 结果图像标签
        result_label = QLabel("分割结果")
        result_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 500;
            color: {COLORS['text_primary']};
            padding: 5px;
            background-color: {COLORS['card_bg']};
            border-radius: 4px;
        """)
        result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_label)
        
        # 结果图像画布
        self.result_canvas = ResultCanvas()
        self.result_canvas.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border-radius: 5px;
        """)
        result_layout.addWidget(self.result_canvas)
        
        # 添加到分割器
        splitter.addWidget(original_container)
        splitter.addWidget(result_container)
        
        # 设置初始分割位置
        splitter.setSizes([700, 700])
        
        # 添加分割器到画布布局
        canvas_layout.addWidget(splitter)
        
        # 添加到父布局
        parent_layout.addWidget(canvas_container, 1)  # 1是拉伸因子
    
    def load_image(self):
        """加载图像"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", options=options
        )
        
        if file_path:
            logger.info(f"Selected image file: {file_path}")
            if self.image_canvas.load_image(file_path):
                self.statusBar().showMessage(f'已加载图像: {os.path.basename(file_path)}')
                # 清除结果画布
                self.result_canvas.set_result(None)
    
    def load_model(self):
        """加载SAM2模型"""
        try:
            # 选择模型大小
            model_size = self.select_model_size()
            if not model_size:
                self.statusBar().showMessage('模型加载取消')
                return
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # 检查本地模型文件
            model_info = MODEL_INFO[model_size]
            model_dir = get_model_dir()
            model_path = os.path.join(model_dir, model_info["file_name"])
            
            # 检查本地文件是否存在且有效（非0字节）
            valid_local_file = os.path.exists(model_path) and os.path.getsize(model_path) > 0
            
            if not valid_local_file:
                reply = QMessageBox.question(
                    self, 
                    "下载模型", 
                    f"未找到本地模型文件，需要下载模型 ({model_info['name']})。\n"
                    f"这可能需要几分钟时间，是否继续？\n"
                    f"（文件将保存至: {model_dir}）",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.statusBar().showMessage(f'正在下载模型: {model_info["name"]}...')
                    # 下载模型
                    if not self._download_model(model_info, model_path):
                        self.progress_bar.setVisible(False)
                        return
                else:
                    self.statusBar().showMessage('取消模型加载')
                    self.progress_bar.setVisible(False)
                    return
            
            # 加载模型
            self.statusBar().showMessage('正在加载模型...')
            self.progress_bar.setValue(50)
            QApplication.processEvents()  # 更新UI
            
            logger.info(f"Loading SAM2 model from: {model_path}")
            
            # 尝试使用不同的加载方法
            load_success = False
            error_messages = []
            
            # 方法1: 直接从本地路径加载
            try:
                self.predictor = SAM2ImagePredictor.from_checkpoint(model_path)
                logger.info("SAM2 model loaded successfully from local file")
                self.statusBar().showMessage('模型加载成功')
                self.progress_bar.setValue(100)
                load_success = True
                QMessageBox.information(self, "成功", f"SAM2 模型 ({model_info['name']}) 已成功加载！")
            except Exception as local_err:
                error_msg = str(local_err)
                logger.error(f"Error loading local model: {error_msg}")
                error_messages.append(f"本地加载失败: {error_msg}")
                self.progress_bar.setValue(60)
                
                # 方法2: 尝试从Hugging Face加载
                if not load_success:
                    try:
                        logger.info(f"Attempting to load model from Hugging Face: {model_info['repo']}")
                        self.statusBar().showMessage('尝试从Hugging Face加载模型...')
                        self.predictor = SAM2ImagePredictor.from_pretrained(model_info['repo'])
                        logger.info("SAM2 model loaded successfully from Hugging Face")
                        self.statusBar().showMessage('模型加载成功')
                        self.progress_bar.setValue(100)
                        load_success = True
                        QMessageBox.information(self, "成功", f"SAM2 模型 ({model_info['name']}) 已成功加载！")
                    except Exception as remote_err:
                        error_msg = str(remote_err)
                        logger.error(f"Error loading model from Hugging Face: {error_msg}")
                        error_messages.append(f"在线加载失败: {error_msg}")
                        self.progress_bar.setValue(70)
            
            # 如果所有加载方法都失败
            if not load_success:
                error_details = "\n".join(error_messages)
                
                # 提供更友好的错误消息
                if any("Max retries exceeded" in msg or "SSLError" in msg for msg in error_messages):
                    QMessageBox.critical(
                        self, 
                        "网络错误", 
                        "无法连接到Hugging Face服务器，可能是网络问题。\n\n"
                        "建议：\n"
                        "1. 检查您的网络连接\n"
                        "2. 检查是否需要配置代理\n"
                        "3. 尝试使用VPN连接\n"
                        "4. 手动下载模型文件并放置到以下目录：\n"
                        f"{model_dir}\n\n"
                        f"错误详情:\n{error_details}"
                    )
                else:
                    QMessageBox.critical(
                        self, 
                        "错误", 
                        f"无法加载模型，请尝试重新下载。\n\n错误详情:\n{error_details}"
                    )
                self.statusBar().showMessage('模型加载失败')
            
            self.progress_bar.setVisible(False)
                
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载模型过程中出错: {str(e)}")
            self.statusBar().showMessage('模型加载失败')
            self.progress_bar.setVisible(False)
    
    def _download_model(self, model_info, save_path):
        """下载模型文件，带进度条和重试机制"""
        # 创建进度对话框
        progress = QProgressDialog("正在下载模型...", "取消", 0, 100, self)
        progress.setWindowTitle("下载模型")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setStyleSheet(f"""
            QProgressDialog {{
                background-color: {COLORS['card_bg']};
                border-radius: 10px;
                min-width: 400px;
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 14px;
                margin-bottom: 10px;
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
            }}
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {COLORS['divider']};
                height: 10px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
        """)
        progress.show()
        
        # 使用更可靠的下载方法
        max_retries = 5  # 增加重试次数
        retry_count = 0
        auto_retry_count = 0
        max_auto_retries = 2  # 自动重试次数
        timeout_seconds = 60  # 增加超时时间
        chunk_size = 8192  # 下载块大小
        
        # 尝试多个下载URL，包括官方和备用镜像
        download_urls = [
            model_info['download_url'],
            f"https://hf-mirror.com/{model_info['repo']}/resolve/main/{model_info['file_name']}",  # 镜像1
            f"https://huggingface.co/{model_info['repo']}/resolve/main/{model_info['file_name']}?download=true"  # 强制下载
        ]
        
        # 创建临时文件，用于断点续传
        temp_file = f"{save_path}.part"
        
        while retry_count < max_retries:
            try:
                # 获取已下载的大小，用于断点续传
                existing_size = 0
                if os.path.exists(temp_file):
                    existing_size = os.path.getsize(temp_file)
                    logger.info(f"Resuming download from {existing_size} bytes")
                    progress.setLabelText(f"正在继续下载模型...(已下载 {existing_size/1024/1024:.1f}MB)")
                
                # 创建会话，进行下载
                session = requests.Session()
                
                # 禁用SSL警告
                urllib3.disable_warnings()
                
                # 选择下载URL，每次重试使用不同的URL
                url_index = min(retry_count, len(download_urls) - 1)
                download_url = download_urls[url_index]
                logger.info(f"Trying download URL {url_index + 1}/{len(download_urls)}: {download_url}")
                progress.setLabelText(f"正在尝试下载源 {url_index + 1}: {os.path.basename(download_url)}")
                QApplication.processEvents()
                
                # 设置请求头，用于断点续传
                headers = {}
                if existing_size > 0:
                    headers['Range'] = f'bytes={existing_size}-'
                
                # 设置较长的超时时间
                response = session.get(
                    download_url, 
                    stream=True, 
                    timeout=timeout_seconds,
                    verify=True,  # 保持SSL验证启用
                    headers=headers
                )
                response.raise_for_status()
                
                # 检查是否支持断点续传
                if existing_size > 0 and response.status_code != 206:
                    logger.warning("Server doesn't support resume, starting from beginning")
                    existing_size = 0
                
                # 获取文件大小
                if 'content-length' in response.headers:
                    content_length = int(response.headers['content-length'])
                    total_size = existing_size + content_length
                else:
                    # 如果无法获取大小，使用一个估计值
                    content_length = 0
                    total_size = 500000000  # 假设500MB
                
                logger.info(f"Download started, total size: {total_size/1024/1024:.1f}MB")
                
                # 打开文件用于写入
                mode = 'ab' if existing_size > 0 else 'wb'
                with open(temp_file, mode) as f:
                    downloaded = existing_size
                    start_time = time.time()
                    last_update_time = start_time
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if progress.wasCanceled():
                            f.close()
                            logger.info("Download cancelled by user")
                            self.statusBar().showMessage('取消下载')
                            return False
                        
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 更新进度
                            now = time.time()
                            if now - last_update_time > 0.1:  # 每0.1秒更新一次UI
                                percent = int(downloaded / total_size * 100) if total_size > 0 else 0
                                speed = downloaded / (now - start_time) / 1024  # KB/s
                                
                                if speed > 1024:
                                    speed_text = f"{speed/1024:.1f} MB/s"
                                else:
                                    speed_text = f"{speed:.1f} KB/s"
                                
                                progress.setLabelText(f"正在下载模型... {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({speed_text})")
                                progress.setValue(min(percent, 100))
                                QApplication.processEvents()
                                last_update_time = now
                
                # 下载完成，重命名临时文件
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(temp_file, save_path)
                
                progress.setValue(100)
                logger.info(f"Model downloaded successfully to: {save_path}")
                return True
                
            except (requests.RequestException, IOError) as e:
                error_msg = str(e)
                logger.error(f"Download attempt {retry_count + 1} failed: {error_msg}")
                
                # 自动重试几次，不打扰用户
                if auto_retry_count < max_auto_retries:
                    auto_retry_count += 1
                    logger.info(f"Auto-retrying ({auto_retry_count}/{max_auto_retries})...")
                    progress.setLabelText(f"下载出错，正在自动重试 ({auto_retry_count}/{max_auto_retries})...")
                    progress.setValue(0)
                    QApplication.processEvents()
                    time.sleep(2)  # 等待2秒后重试
                    continue
                
                # 超过自动重试次数，询问用户
                retry_count += 1
                if retry_count < max_retries:
                    retry_msg = (
                        f"下载失败 (尝试 {retry_count}/{max_retries}):\n"
                        f"{error_msg}\n\n"
                        "是否更换下载源重试？"
                    )
                    retry = QMessageBox.question(
                        self, "下载错误", retry_msg, 
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if retry == QMessageBox.Yes:
                        auto_retry_count = 0  # 重置自动重试计数
                        progress.setValue(0)
                        continue
                    else:
                        break
                else:
                    # 达到最大重试次数
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
                    QMessageBox.critical(
                        self, 
                        "下载失败", 
                        f"多次尝试后仍无法下载模型:\n{error_msg}\n\n"
                        "可能的解决方法:\n"
                        "1. 检查网络连接\n"
                        "2. 尝试使用VPN\n"
                        "3. 手动下载模型:\n"
                        f"   - 网址: {model_info['download_url']}\n"
                        f"   - 保存至: {save_path}\n\n"
                        "是否尝试使用备用下载方法？"
                    )
                    
                    # 提供备用下载选项
                    backup_options = QMessageBox(self)
                    backup_options.setWindowTitle("备用下载选项")
                    backup_options.setText("请选择备用下载方式:")
                    backup_options.setIcon(QMessageBox.Question)
                    
                    browser_btn = backup_options.addButton("使用浏览器下载", QMessageBox.ActionRole)
                    skip_btn = backup_options.addButton("跳过下载", QMessageBox.ActionRole)
                    cancel_btn = backup_options.addButton("取消", QMessageBox.RejectRole)
                    
                    backup_options.exec_()
                    
                    clicked_button = backup_options.clickedButton()
                    if clicked_button == browser_btn:
                        # 打开浏览器下载页面
                        import webbrowser
                        download_page = f"https://huggingface.co/{model_info['repo']}/blob/main/{model_info['file_name']}"
                        webbrowser.open(download_page)
                        QMessageBox.information(
                            self,
                            "手动下载指南",
                            f"1. 在打开的网页中，点击'下载'按钮\n"
                            f"2. 将下载的文件移动到以下位置:\n{save_path}\n"
                            f"3. 然后重新启动应用程序"
                        )
                        return False
                    elif clicked_button == skip_btn:
                        return False
                    else:
                        return False
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        progress.close()
        return False
    
    def select_model_size(self):
        """选择模型大小"""
        # 创建选择对话框
        dialog = QMessageBox(self)
        dialog.setWindowTitle("选择模型大小")
        dialog.setText("请选择要加载的SAM2模型大小：\n(根据您的设备性能选择合适的大小)")
        dialog.setIcon(QMessageBox.Question)
        
        # 添加按钮
        buttons = {}
        for size, info in MODEL_INFO.items():
            button = dialog.addButton(f"{size} - {info['description']}", QMessageBox.ActionRole)
            buttons[button] = size
        
        # 添加CPU模式选项
        cpu_button = dialog.addButton("CPU模式 (使用tiny模型)", QMessageBox.ActionRole)
        buttons[cpu_button] = "tiny"
        
        cancel_button = dialog.addButton("取消", QMessageBox.RejectRole)
        
        # 显示对话框并获取结果
        dialog.exec_()
        
        clicked_button = dialog.clickedButton()
        if clicked_button == cancel_button:
            return None
        
        return buttons[clicked_button]
    
    def increase_obj_id(self):
        """增加对象ID"""
        self.current_obj_id += 1
        self.update_obj_id_label()
        
    def decrease_obj_id(self):
        """减少对象ID"""
        if self.current_obj_id > 1:
            self.current_obj_id -= 1
            self.update_obj_id_label()
            
    def update_obj_id_label(self):
        """更新对象ID标签"""
        self.obj_id_label.setText(f"{self.current_obj_id}")
    
    def toggle_prompt_type(self):
        """切换提示类型"""
        is_point_mode = self.point_radio.isChecked()
        is_box_mode = self.box_radio.isChecked()
        is_semantic_mode = self.semantic_radio.isChecked()
        
        # 更新画布模式
        self.image_canvas.set_point_mode(is_point_mode)
        
        # 显示/隐藏相应的选项
        self.point_type_container.setVisible(is_point_mode)
        self.semantic_options_container.setVisible(is_semantic_mode)
        
        # 更新状态栏提示
        if is_point_mode:
            self.statusBar().showMessage('点标注模式：在图像上点击添加前景/背景点')
        elif is_box_mode:
            self.statusBar().showMessage('框标注模式：拖动鼠标绘制选择框')
        elif is_semantic_mode:
            self.statusBar().showMessage('语义分割模式：将自动分割图像中的所有物体')
    
    def update_threshold_value(self, value):
        """更新阈值显示值"""
        self.threshold_value_label.setText(f"{value}%")
    
    def toggle_point_type(self):
        """切换点类型"""
        is_foreground = self.foreground_radio.isChecked()
        self.image_canvas.set_foreground_point(is_foreground)
    
    def clear_prompts(self):
        """清除所有提示"""
        self.image_canvas.clear_prompts()
        self.statusBar().showMessage('已清除所有提示')
    
    def perform_segmentation(self):
        """执行分割"""
        if self.predictor is None:
            logger.warning("Segmentation attempted without loading model")
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if self.image_canvas.image is None:
            logger.warning("Segmentation attempted without loading image")
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        # 检查是否为语义分割模式
        is_semantic_mode = self.semantic_radio.isChecked()
        
        if is_semantic_mode:
            # 执行语义分割
            self.perform_semantic_segmentation()
            return
        
        # 获取提示数据
        point_coords, point_labels, box = self.image_canvas.get_prompt_data()
        
        if point_coords is None and box is None:
            logger.warning("Segmentation attempted without prompts")
            QMessageBox.warning(self, "警告", "请添加至少一个点或一个框")
            return
        
        try:
            self.statusBar().showMessage('正在分割...')
            QApplication.processEvents()  # 更新UI
            
            logger.info(f"Performing segmentation for object ID: {self.current_obj_id}")
            # 执行分割
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(self.image_canvas.image)
                masks, iou_scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True
                )
            
            # 获取最佳掩码（IoU最高的）
            best_mask_idx = np.argmax(iou_scores)
            best_mask = masks[best_mask_idx]
            best_iou = iou_scores[best_mask_idx]
            logger.info(f"Segmentation complete, best IoU: {best_iou:.4f} for object ID: {self.current_obj_id}")
            
            # 保存当前掩码
            self.current_mask = best_mask
            
            # 创建结果图像，使用对象ID作为颜色的变化
            color_options = [
                (240, 100, 200),  # 粉色
                (100, 200, 100),  # 绿色
                (100, 100, 240),  # 蓝色
                (240, 200, 100),  # 橙色
                (180, 100, 180)   # 紫色
            ]
            color_idx = (self.current_obj_id - 1) % len(color_options)
            color = color_options[color_idx]
            
            # 创建结果图像
            result_img = self.visualize_segmentation(self.image_canvas.image, best_mask, color=color)
            
            # 创建掩码可视化图像
            mask_img = self.visualize_mask(best_mask, self.image_canvas.image.size, color=color)
            
            # 显示结果
            self.result_canvas.set_result(result_img)
            
            # 显示一个对话框，询问用户是否要查看掩码
            reply = QMessageBox.question(self, '查看掩码', 
                                        f'对象 {self.current_obj_id} 分割完成，是否查看掩码？',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                self.result_canvas.set_result(mask_img)
                self.statusBar().showMessage(f'显示对象 {self.current_obj_id} 掩码，IoU: {best_iou:.4f}')
            else:
                self.statusBar().showMessage(f'对象 {self.current_obj_id} 分割完成，IoU: {best_iou:.4f}')
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")
            self.statusBar().showMessage('分割失败')
    
    def perform_semantic_segmentation(self):
        """执行语义分割"""
        try:
            self.statusBar().showMessage('正在执行语义分割...')
            QApplication.processEvents()  # 更新UI
            
            # 创建进度对话框
            progress = QProgressDialog("正在执行语义分割...", "取消", 0, 100, self)
            progress.setWindowTitle("语义分割")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            progress.setStyleSheet(f"""
                QProgressDialog {{
                    background-color: {COLORS['card_bg']};
                    border-radius: 10px;
                    min-width: 400px;
                }}
                QLabel {{
                    color: {COLORS['text_primary']};
                    font-size: 14px;
                    margin-bottom: 10px;
                }}
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: 500;
                }}
            """)
            progress.show()
            QApplication.processEvents()
            
            # 设置图像
            logger.info("Setting image for semantic segmentation")
            self.predictor.set_image(self.image_canvas.image)
            progress.setValue(30)
            QApplication.processEvents()
            
            # 获取分割阈值
            threshold = self.threshold_slider.value() / 100.0
            logger.info(f"Performing semantic segmentation with threshold: {threshold}")
            
            # 执行语义分割 - 使用SAM的自动分割模式
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # 生成图像嵌入
                progress.setLabelText("生成图像特征...")
                progress.setValue(40)
                QApplication.processEvents()
                
                # 使用SAM的自动分割模式，无需提供点或框
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=None,
                    multimask_output=True
                )
                
                progress.setValue(70)
                progress.setLabelText("处理分割结果...")
                QApplication.processEvents()
                
                # 过滤掩码，只保留高于阈值的
                valid_masks = []
                valid_scores = []
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score > threshold:
                        valid_masks.append(mask)
                        valid_scores.append(score)
                
                # 如果没有找到有效掩码
                if not valid_masks:
                    progress.close()
                    QMessageBox.information(self, "结果", "未找到符合阈值的分割区域，请尝试降低阈值")
                    self.statusBar().showMessage('语义分割完成，未找到有效区域')
                    return
                
                # 创建彩色分割结果
                result_img = self.visualize_semantic_segmentation(self.image_canvas.image, valid_masks)
                
                progress.setValue(100)
                QApplication.processEvents()
            
            # 显示结果
            self.result_canvas.set_result(result_img)
            self.statusBar().showMessage(f'语义分割完成，找到 {len(valid_masks)} 个区域')
            
            # 关闭进度对话框
            progress.close()
            
            # 保存当前掩码（使用得分最高的）
            if valid_masks:
                self.current_mask = valid_masks[0]
            
        except Exception as e:
            logger.error(f"Semantic segmentation error: {str(e)}")
            QMessageBox.critical(self, "错误", f"语义分割失败: {str(e)}")
            self.statusBar().showMessage('语义分割失败')
    
    def visualize_segmentation(self, image, mask, alpha=0.7, color=(240, 100, 200)):  # 添加颜色参数
        """可视化分割结果"""
        # 创建图像副本
        result = image.copy()
        
        # 创建彩色掩码
        colored_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask_np = mask.astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np).resize(image.size)
        
        # 为掩码上色 - 使用传入的颜色
        mask_color = (*color, int(255 * alpha))
        
        # 绘制掩码并添加边缘线以增强可见性
        for y in range(mask_pil.height):
            for x in range(mask_pil.width):
                if mask_pil.getpixel((x, y)) > 128:
                    colored_mask.putpixel((x, y), mask_color)
        
        # 将掩码与原图合并
        result = Image.alpha_composite(result.convert("RGBA"), colored_mask)
        
        # 添加边缘线以增强可见性
        edge_mask = self._create_edge_mask(mask_np)
        edge_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for y in range(edge_mask.shape[0]):
            for x in range(edge_mask.shape[1]):
                if edge_mask[y, x]:
                    edge_overlay.putpixel((x, y), (color[0], color[1]//2, color[2]//2, 230))  # 边缘线颜色
        
        # 将边缘线叠加到结果上
        result = Image.alpha_composite(result, edge_overlay)
        
        return result
    
    def _create_edge_mask(self, mask, thickness=2):
        """创建掩码边缘"""
        from scipy import ndimage
        
        # 使用腐蚀和膨胀操作创建边缘
        eroded = ndimage.binary_erosion(mask, iterations=thickness)
        edge_mask = mask.copy()
        edge_mask[eroded > 0] = 0
        
        return edge_mask
    
    def visualize_mask(self, mask, image_size, color=(240, 100, 200)):  # 添加颜色参数
        """可视化掩码"""
        if mask is None:
            return None
            
        # 创建掩码图像
        mask_np = mask.astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np).resize(image_size)
        
        # 创建彩色掩码
        colored_mask = Image.new("RGB", image_size, (0, 0, 0))
        
        # 为掩码上色 - 使用传入的颜色
        mask_color = color
        
        # 绘制掩码
        for y in range(mask_pil.height):
            for x in range(mask_pil.width):
                if mask_pil.getpixel((x, y)) > 128:
                    colored_mask.putpixel((x, y), mask_color)
        
        # 添加边缘线以增强可见性
        edge_mask = self._create_edge_mask(mask_np)
        for y in range(edge_mask.shape[0]):
            for x in range(edge_mask.shape[1]):
                if edge_mask[y, x]:
                    colored_mask.putpixel((x, y), (255, 255, 255))  # 白色边缘
        
        return colored_mask
    
    def toggle_result_view(self):
        """在分割结果和掩码之间切换"""
        # 检查是否有掩码和图像
        if not hasattr(self, 'current_mask') or self.current_mask is None:
            QMessageBox.information(self, "提示", "请先执行分割操作生成掩码")
            return
            
        if self.image_canvas.image is None:
            QMessageBox.information(self, "提示", "请先加载图像")
            return
            
        # 检查当前显示的是否是掩码
        is_showing_mask = False
        if self.result_canvas.result_image:
            try:
                # 更可靠的检查方法 - 尝试获取左上角像素颜色
                left_top_pixel = self.result_canvas.result_image.getpixel((0, 0))
                # 判断是否为掩码图像 - 掩码图像通常有大面积的黑色区域
                is_showing_mask = sum(left_top_pixel[:3]) < 100  # 判断是否为暗色
            except Exception as e:
                logger.error(f"Error checking result image: {str(e)}")
                # 如果检查出错，假设不是掩码
                is_showing_mask = False
        
        try:
            if is_showing_mask:
                # 如果当前显示的是掩码，则切换到分割结果
                logger.info("Switching from mask to segmentation result")
                result_img = self.visualize_segmentation(self.image_canvas.image, self.current_mask)
                self.result_canvas.set_result(result_img)
                self.statusBar().showMessage('显示分割结果')
            else:
                # 如果当前显示的是分割结果，则切换到掩码
                logger.info("Switching from segmentation result to mask")
                mask_img = self.visualize_mask(self.current_mask, self.image_canvas.image.size)
                self.result_canvas.set_result(mask_img)
                self.statusBar().showMessage('显示掩码')
        except Exception as e:
            logger.error(f"Error toggling view: {str(e)}")
            QMessageBox.warning(self, "警告", f"切换视图失败: {str(e)}")
    
    def save_result(self):
        """保存分割结果"""
        if self.result_canvas.result_image is None:
            logger.warning("Save result attempted without result image")
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)", options=options
        )
        
        if file_path:
            try:
                logger.info(f"Saving result to: {file_path}")
                self.result_canvas.result_image.save(file_path)
                self.statusBar().showMessage(f'结果已保存至: {file_path}')
            except Exception as e:
                logger.error(f"Error saving result: {str(e)}")
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def batch_process(self):
        """批量处理文件夹中的图像"""
        if self.predictor is None:
            logger.warning("Batch processing attempted without loading model")
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        # 选择输入文件夹
        input_dir = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if not input_dir:
            return
            
        # 选择输出文件夹
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not output_dir:
            return
            
        # 获取分割方式和参数
        segmentation_options = self.get_batch_segmentation_options()
        if not segmentation_options:
            return
            
        # 查找文件夹中所有支持的图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
            
        if not image_files:
            QMessageBox.warning(self, "警告", "所选文件夹中没有支持的图像文件")
            return
            
        # 创建进度对话框
        progress = QProgressDialog(f"正在处理 0/{len(image_files)} 张图像...", "取消", 0, len(image_files), self)
        progress.setWindowTitle("批量处理")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setStyleSheet(f"""
            QProgressDialog {{
                background-color: {COLORS['card_bg']};
                border-radius: 10px;
                min-width: 400px;
            }}
        """)
        progress.show()
        
        # 计数器
        processed = 0
        failed = 0
        
        # 获取输出格式
        output_format = segmentation_options.get("output_format", "png")
        
        # 开始处理
        for i, image_file in enumerate(image_files):
            if progress.wasCanceled():
                break
                
            try:
                # 更新进度
                progress.setLabelText(f"正在处理 {i+1}/{len(image_files)}: {image_file}")
                progress.setValue(i)
                QApplication.processEvents()
                
                # 完整文件路径
                input_path = os.path.join(input_dir, image_file)
                
                # 加载图像
                image = Image.open(input_path).convert("RGB")
                
                # 执行分割
                result_image = self.process_single_image(image, segmentation_options)
                
                if result_image:
                    # 确定输出路径
                    filename, ext = os.path.splitext(image_file)
                    output_path = os.path.join(output_dir, f"{filename}_segmented.{output_format}")
                    
                    # 保存结果
                    result_image.save(output_path, quality=95 if output_format == 'jpg' else None)
                    processed += 1
                    
                    # 如果需要，保存掩码图像
                    if segmentation_options["save_mask"] and hasattr(self, 'current_mask') and self.current_mask is not None:
                        mask_path = os.path.join(output_dir, f"{filename}_mask.{output_format}")
                        mask_img = self.visualize_mask(self.current_mask, image.size, color=(240, 100, 200))
                        mask_img.save(mask_path, quality=95 if output_format == 'jpg' else None)
                        
                        # 如果需要，保存二值掩码
                        if segmentation_options.get("binary_mask", False):
                            binary_mask_path = os.path.join(output_dir, f"{filename}_binary_mask.png")
                            binary_mask = Image.fromarray((self.current_mask * 255).astype(np.uint8))
                            binary_mask.save(binary_mask_path)
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                failed += 1
                continue
                
        # 处理完成
        progress.setValue(len(image_files))
        QMessageBox.information(
            self, 
            "批量处理完成", 
            f"处理完成！\n成功: {processed} 张图像\n失败: {failed} 张图像\n\n结果已保存至: {output_dir}"
        )
    
    def get_batch_segmentation_options(self):
        """获取批量分割的选项"""
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("批量处理选项")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['card_bg']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QComboBox {{
                padding: 8px;
                border: 1px solid {COLORS['divider']};
                border-radius: 4px;
                background-color: {COLORS['background']};
                min-height: 30px;
            }}
            QCheckBox {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
        """)
        
        layout = QFormLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 分割模式选择
        mode_combo = QComboBox()
        mode_combo.addItem("自动语义分割", "semantic")
        mode_combo.addItem("中心点分割", "center_point")
        mode_combo.addItem("自动框分割", "auto_box")
        layout.addRow("分割模式:", mode_combo)
        
        # 阈值选择（仅用于语义分割）
        threshold_combo = QComboBox()
        threshold_combo.addItem("低 (30%)", 0.3)
        threshold_combo.addItem("中 (50%)", 0.5)
        threshold_combo.addItem("高 (70%)", 0.7)
        layout.addRow("分割阈值:", threshold_combo)
        
        # 输出格式选择
        format_combo = QComboBox()
        format_combo.addItem("PNG (无损)", "png")
        format_combo.addItem("JPEG (高质量)", "jpg")
        format_combo.addItem("WEBP (高压缩)", "webp")
        layout.addRow("输出格式:", format_combo)
        
        # 保存原始掩码选项
        save_mask_checkbox = QCheckBox("同时保存掩码图像")
        save_mask_checkbox.setChecked(True)
        layout.addRow("", save_mask_checkbox)
        
        # 添加二值掩码选项
        binary_mask_checkbox = QCheckBox("保存二值掩码 (黑白掩码)")
        binary_mask_checkbox.setChecked(False)
        layout.addRow("", binary_mask_checkbox)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(f"""
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        """)
        layout.addRow(buttons)
        
        # 执行对话框
        if dialog.exec_() == QDialog.Accepted:
            return {
                "mode": mode_combo.currentData(),
                "threshold": threshold_combo.currentData(),
                "save_mask": save_mask_checkbox.isChecked(),
                "binary_mask": binary_mask_checkbox.isChecked(),
                "output_format": format_combo.currentData()
            }
        return None
    
    def process_single_image(self, image, options):
        """处理单张图像并返回结果"""
        try:
            # 设置图像
            self.predictor.set_image(image)
            
            mode = options["mode"]
            threshold = options["threshold"]
            
            # 根据选择的模式执行分割
            if mode == "semantic":
                # 语义分割模式
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=None,
                        multimask_output=True
                    )
                    
                    # 过滤掩码，只保留高于阈值的
                    valid_masks = []
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        if score > threshold:
                            valid_masks.append(mask)
                    
                    if valid_masks:
                        # 保存最好的掩码供保存使用
                        self.current_mask = valid_masks[0]
                        # 创建彩色分割结果
                        return self.visualize_semantic_segmentation(image, valid_masks)
                    else:
                        self.current_mask = None
                        return None
                        
            elif mode == "center_point":
                # 中心点分割 - 在图像中心添加一个前景点
                height, width = image.height, image.width
                center_x, center_y = width // 2, height // 2
                
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    point_coords = np.array([[center_x, center_y]])
                    point_labels = np.array([1])  # 1表示前景
                    
                    masks, iou_scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=None,
                        multimask_output=True
                    )
                    
                    # 获取最佳掩码
                    best_mask_idx = np.argmax(iou_scores)
                    best_mask = masks[best_mask_idx]
                    
                    # 保存掩码供保存使用
                    self.current_mask = best_mask
                    
                    # 创建结果图像
                    return self.visualize_segmentation(image, best_mask, color=(240, 100, 200))
                    
            elif mode == "auto_box":
                # 自动框分割 - 创建一个覆盖图像中心80%区域的框
                height, width = image.height, image.width
                margin_x = int(width * 0.1)
                margin_y = int(height * 0.1)
                box = np.array([margin_x, margin_y, width - margin_x, height - margin_y])
                
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, iou_scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=True
                    )
                    
                    # 获取最佳掩码
                    best_mask_idx = np.argmax(iou_scores)
                    best_mask = masks[best_mask_idx]
                    
                    # 保存掩码供保存使用
                    self.current_mask = best_mask
                    
                    # 创建结果图像
                    return self.visualize_segmentation(image, best_mask, color=(100, 200, 100))
            
            self.current_mask = None
            return None
            
        except Exception as e:
            logger.error(f"Error in process_single_image: {str(e)}")
            return None
    
    def visualize_semantic_segmentation(self, image, masks):
        """可视化语义分割结果，为不同区域使用不同颜色"""
        # 创建图像副本
        result = image.copy().convert("RGBA")
        
        # 颜色列表，用于区分不同区域
        colors = [
            (240, 100, 200, 160),  # 粉色
            (100, 200, 100, 160),  # 绿色
            (100, 100, 240, 160),  # 蓝色
            (240, 200, 100, 160),  # 橙色
            (180, 100, 180, 160),  # 紫色
            (100, 200, 200, 160),  # 青色
            (200, 150, 100, 160),  # 棕色
            (150, 150, 240, 160),  # 淡蓝
            (240, 150, 150, 160),  # 淡红
            (180, 220, 100, 160)   # 黄绿
        ]
        
        # 创建彩色掩码
        colored_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(colored_mask)
        
        # 为每个掩码上色
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_np = mask.astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np).resize(image.size)
            
            # 创建此掩码的彩色版本
            mask_color = Image.new("RGBA", image.size, (0, 0, 0, 0))
            for y in range(mask_pil.height):
                for x in range(mask_pil.width):
                    if mask_pil.getpixel((x, y)) > 128:
                        mask_color.putpixel((x, y), color)
            
            # 添加到总掩码
            colored_mask = Image.alpha_composite(colored_mask, mask_color)
            
            # 添加边缘线
            edge_mask = self._create_edge_mask(mask_np)
            edge_color = (color[0], color[1], color[2], 230)  # 边缘线颜色
            for y in range(edge_mask.shape[0]):
                for x in range(edge_mask.shape[1]):
                    if edge_mask[y, x]:
                        colored_mask.putpixel((x, y), edge_color)
        
        # 将掩码与原图合并
        result = Image.alpha_composite(result, colored_mask)
        
        return result


if __name__ == "__main__":
    try:
        logger.info("Starting SAM2 UI application")
        app = QApplication(sys.argv)
        
        # 设置应用级别的异常处理
        def exception_hook(exctype, value, traceback_obj):
            """全局异常钩子，捕获未处理的异常"""
            logger.critical("Uncaught exception", exc_info=(exctype, value, traceback_obj))
            msg = f"发生未捕获的异常:\n{exctype.__name__}: {value}"
            QMessageBox.critical(None, "错误", msg)
        
        # 替换默认的异常处理器
        sys.excepthook = exception_hook
        
        ui = SAM2UI()
        ui.show()
        logger.info("Application window shown")
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        print(f"Critical error: {str(e)}")
        QMessageBox.critical(None, "严重错误", f"应用程序崩溃: {str(e)}")
        sys.exit(1) 