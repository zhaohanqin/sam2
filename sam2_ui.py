import sys
import os
import numpy as np
import torch
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup,
                            QSlider, QGroupBox, QMessageBox, QSplitter)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QRect, QPoint
from PIL import Image, ImageDraw
from sam2.sam2_image_predictor import SAM2ImagePredictor
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量
COLORS = {
    "background": "#FFFFFF",  # 背景色，改为纯白色
    "foreground": "#F5F5F5",  # 前景色，改为浅灰色
    "accent1": "#E1F5FE",     # 强调色1，改为浅蓝色
    "accent2": "#F3E5F5"      # 强调色2，改为浅紫色
}

# 设置全局字体样式
FONT_FAMILY = "微软雅黑"  # 更圆润的字体
FONT_SIZE = 12  # 更大的字体尺寸
BUTTON_FONT_SIZE = 16  # 增大按钮字体尺寸
GROUP_TITLE_FONT_SIZE = 16  # 新增：选项组标题字体大小
RADIO_BUTTON_FONT_SIZE = 14  # 新增：单选按钮字体大小

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
        self.setMinimumSize(600, 600)  # 增大最小尺寸
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
        self.setMinimumSize(600, 600)  # 增大最小尺寸
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
            logger.info("Initializing SAM2UI")
            self.initUI()
        except Exception as e:
            logger.critical(f"Failed to initialize UI: {str(e)}", exc_info=True)
            QMessageBox.critical(None, "严重错误", f"初始化界面失败: {str(e)}")
            sys.exit(1)
    
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('SAM2 交互式分割')
        self.setGeometry(100, 100, 1400, 900)  # 增大窗口尺寸
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QGroupBox {{
                font-weight: bold;
            }}
            QLabel {{
                color: #333333;
            }}
        """)
        
        # 设置应用字体
        app_font = QFont(FONT_FAMILY, FONT_SIZE)
        app_font.setStyleStrategy(QFont.PreferAntialias)  # 使字体更平滑
        QApplication.setFont(app_font)
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # 增加布局间距
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 创建顶部工具栏
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(15, 15, 15, 15)  # 增加边距
        toolbar_layout.setSpacing(15)  # 增加间距
        
        # 统一按钮样式
        button_style = f"""
            QPushButton {{
                background-color: {COLORS['foreground']};
                color: #333333;
                border: 2px solid {COLORS['accent2']};
                border-radius: 8px;
                padding: 12px;
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent2']};
                border-color: {COLORS['accent1']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent1']};
                border-color: {COLORS['accent2']};
            }}
        """
        
        # 加载图像按钮
        load_btn = QPushButton("加载图像")
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet(button_style)
        load_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(load_btn)
        
        # 加载模型按钮
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        load_model_btn.setStyleSheet(button_style)
        load_model_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(load_model_btn)
        
        # 清除提示按钮
        clear_btn = QPushButton("清除提示")
        clear_btn.clicked.connect(self.clear_prompts)
        clear_btn.setStyleSheet(button_style)
        clear_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(clear_btn)
        
        # 分割按钮
        segment_btn = QPushButton("分割")
        segment_btn.clicked.connect(self.perform_segmentation)
        segment_btn.setStyleSheet(button_style)
        segment_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(segment_btn)
        
        # 切换视图按钮
        toggle_view_btn = QPushButton("切换视图")
        toggle_view_btn.clicked.connect(self.toggle_result_view)
        toggle_view_btn.setStyleSheet(button_style)
        toggle_view_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(toggle_view_btn)
        
        # 保存结果按钮
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self.save_result)
        save_btn.setStyleSheet(button_style)
        save_btn.setMinimumHeight(50)  # 增加按钮高度
        toolbar_layout.addWidget(save_btn)
        
        main_layout.addWidget(toolbar_widget)
        
        # 创建选项组 - 使用更紧凑的布局
        options_widget = QWidget()
        options_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
                border-radius: 10px;
                margin: 0px;
            }}
        """)
        options_layout = QHBoxLayout(options_widget)
        options_layout.setContentsMargins(10, 0, 10, 0)  # 减小垂直内边距
        
        # 提示类型和点类型的共同样式
        option_group_style = f"""
            QGroupBox {{
                background-color: {COLORS['accent2']};
                border-radius: 10px;
                padding: 8px;  /* 增加内边距 */
                font-weight: bold;
                margin-top: 25px;  /* 增加顶部边距，为标题腾出空间 */
                max-height: 120px;  /* 增加高度 */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;  /* 增加左边距 */
                top: 15px;  /* 调整标题位置 */
                padding: 0 8px 0 8px;  /* 增加水平内边距 */
                background-color: {COLORS['accent2']};
                color: #333333;
                font-size: {GROUP_TITLE_FONT_SIZE}px;  /* 使用新的标题字体大小 */
                font-weight: bold;  /* 加粗标题 */
            }}
            /* 使单选按钮看起来像按钮 */
            QRadioButton {{
                font-size: {RADIO_BUTTON_FONT_SIZE}px;  /* 使用新的单选按钮字体大小 */
                padding: 8px;  /* 增加内边距 */
                spacing: 8px;  /* 增加间距 */
                min-height: 36px;  /* 增加最小高度 */
                background-color: {COLORS['background']};
                border-radius: 6px;  /* 增加圆角 */
                margin: 4px;  /* 增加外边距 */
                font-weight: bold;  /* 加粗文本 */
            }}
            QRadioButton::indicator {{
                width: 0px;  /* 隐藏默认指示器 */
                height: 0px;
            }}
            QRadioButton:checked {{
                font-weight: bold;
                color: #333333;
                background-color: {COLORS['accent1']};  /* 选中时背景变色 */
                border: 2px solid #333333;
            }}
        """
        
        # 创建水平布局包含两个选项组，让它们占满整个宽度
        compact_options = QHBoxLayout()
        compact_options.setSpacing(30)  # 增加组之间的间距
        
        # 提示类型选项组 - 更紧凑的设计
        prompt_group = QGroupBox("提示类型")
        prompt_group.setStyleSheet(option_group_style)
        prompt_layout = QHBoxLayout()
        prompt_layout.setContentsMargins(8, 15, 8, 8)  # 增加内边距，特别是顶部
        prompt_layout.setSpacing(25)  # 增加间距使按钮占据更多空间
        
        self.point_radio = QRadioButton("点")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(self.toggle_prompt_type)
        
        self.box_radio = QRadioButton("框")
        
        # 设置按钮的伸展策略，使其占满宽度
        prompt_layout.addWidget(self.point_radio, 1)  # 设置拉伸因子为1
        prompt_layout.addWidget(self.box_radio, 1)  # 设置拉伸因子为1
        prompt_group.setLayout(prompt_layout)
        
        # 点类型选项组 - 更紧凑的设计
        point_type_group = QGroupBox("点类型")
        point_type_group.setStyleSheet(option_group_style)
        point_type_layout = QHBoxLayout()
        point_type_layout.setContentsMargins(8, 15, 8, 8)  # 增加内边距，特别是顶部
        point_type_layout.setSpacing(25)  # 增加间距使按钮占据更多空间
        
        self.foreground_radio = QRadioButton("前景")
        self.foreground_radio.setChecked(True)
        self.foreground_radio.toggled.connect(self.toggle_point_type)
        
        self.background_radio = QRadioButton("背景")
        
        # 设置按钮的伸展策略，使其占满宽度
        point_type_layout.addWidget(self.foreground_radio, 1)  # 设置拉伸因子为1
        point_type_layout.addWidget(self.background_radio, 1)  # 设置拉伸因子为1
        point_type_group.setLayout(point_type_layout)
        
        # 将两个组添加到布局中，并设置它们占据相等的宽度
        compact_options.addWidget(prompt_group, 1)  # 设置拉伸因子为1
        compact_options.addWidget(point_type_group, 1)  # 设置拉伸因子为1
        
        # 增加选项组的垂直空间
        options_widget.setMinimumHeight(140)  # 设置最小高度
        options_layout.addLayout(compact_options)
        main_layout.addWidget(options_widget)
        
        # 创建图像显示区域
        display_widget = QWidget()
        display_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
                border-radius: 10px;
            }}
        """)
        display_layout = QHBoxLayout(display_widget)
        display_layout.setContentsMargins(10, 5, 10, 10)  # 减小上边距，增大其他边距
        
        # 创建分割器，允许调整两个画布的大小
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)  # 增加分割条宽度
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['accent2']};
                border-radius: 4px;
            }}
        """)
        
        # 原始图像画布
        self.image_canvas = ImageCanvas()
        splitter.addWidget(self.image_canvas)
        
        # 结果图像画布
        self.result_canvas = ResultCanvas()
        splitter.addWidget(self.result_canvas)
        
        # 设置初始分割位置
        splitter.setSizes([700, 700])
        
        display_layout.addWidget(splitter)
        main_layout.addWidget(display_widget, 1)  # 添加拉伸因子，使图像区域占据更多空间
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        self.statusBar().setStyleSheet(f"""
            background-color: {COLORS['foreground']};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE}px;
            padding: 5px;
        """)
        
        logger.info("UI initialization complete")
    
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
            self.statusBar().showMessage('正在加载模型...')
            QApplication.processEvents()  # 更新UI
            
            logger.info("Loading SAM2 model...")
            # 加载SAM2模型
            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
            logger.info("SAM2 model loaded successfully")
            
            self.statusBar().showMessage('模型加载成功')
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法加载模型: {str(e)}")
            self.statusBar().showMessage('模型加载失败')
    
    def toggle_prompt_type(self):
        """切换提示类型"""
        is_point_mode = self.point_radio.isChecked()
        self.image_canvas.set_point_mode(is_point_mode)
    
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
        
        # 获取提示数据
        point_coords, point_labels, box = self.image_canvas.get_prompt_data()
        
        if point_coords is None and box is None:
            logger.warning("Segmentation attempted without prompts")
            QMessageBox.warning(self, "警告", "请添加至少一个点或一个框")
            return
        
        try:
            self.statusBar().showMessage('正在分割...')
            QApplication.processEvents()  # 更新UI
            
            logger.info("Performing segmentation...")
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
            logger.info(f"Segmentation complete, best IoU: {best_iou:.4f}")
            
            # 保存当前掩码
            self.current_mask = best_mask
            
            # 创建结果图像
            result_img = self.visualize_segmentation(self.image_canvas.image, best_mask)
            
            # 创建掩码可视化图像
            mask_img = self.visualize_mask(best_mask, self.image_canvas.image.size)
            
            # 显示结果
            self.result_canvas.set_result(result_img)
            
            # 显示一个对话框，询问用户是否要查看掩码
            reply = QMessageBox.question(self, '查看掩码', 
                                        '分割完成，是否查看掩码？',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                self.result_canvas.set_result(mask_img)
                self.statusBar().showMessage(f'显示掩码，IoU: {best_iou:.4f}')
            else:
                self.statusBar().showMessage(f'分割完成，IoU: {best_iou:.4f}')
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")
            self.statusBar().showMessage('分割失败')
    
    def visualize_segmentation(self, image, mask, alpha=0.7):  # 增加透明度
        """可视化分割结果"""
        # 创建图像副本
        result = image.copy()
        
        # 创建彩色掩码
        colored_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask_np = mask.astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np).resize(image.size)
        
        # 为掩码上色 - 使用更深的粉色
        mask_color = (240, 100, 200, int(255 * alpha))  # 更深的粉色，增加透明度
        
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
                    edge_overlay.putpixel((x, y), (255, 50, 150, 230))  # 边缘线颜色
        
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
    
    def visualize_mask(self, mask, image_size):
        """可视化掩码"""
        if mask is None:
            return None
            
        # 创建掩码图像
        mask_np = mask.astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np).resize(image_size)
        
        # 创建彩色掩码
        colored_mask = Image.new("RGB", image_size, (0, 0, 0))
        
        # 为掩码上色 - 使用明亮的颜色
        mask_color = (240, 100, 200)  # 粉色
        
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
        if self.current_mask is None or self.image_canvas.image is None:
            return
            
        # 检查当前显示的是否是掩码
        is_showing_mask = False
        if self.result_canvas.result_image:
            # 简单检查：掩码图像的左上角像素通常是黑色的
            left_top_pixel = self.result_canvas.result_image.getpixel((0, 0))
            is_showing_mask = left_top_pixel == (0, 0, 0)
        
        if is_showing_mask:
            # 如果当前显示的是掩码，则切换到分割结果
            result_img = self.visualize_segmentation(self.image_canvas.image, self.current_mask)
            self.result_canvas.set_result(result_img)
            self.statusBar().showMessage('显示分割结果')
        else:
            # 如果当前显示的是分割结果，则切换到掩码
            mask_img = self.visualize_mask(self.current_mask, self.image_canvas.image.size)
            self.result_canvas.set_result(mask_img)
            self.statusBar().showMessage('显示掩码')
    
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