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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量
COLORS = {
    "background": "#C8F5ED",  # 背景色
    "foreground": "#FDFCC6",  # 前景色
    "accent1": "#FED0F3",     # 强调色1
    "accent2": "#D8C2FC"      # 强调色2
}

# 设置全局字体样式
FONT_FAMILY = "微软雅黑"  # 更圆润的字体
FONT_SIZE = 12  # 更大的字体尺寸
BUTTON_FONT_SIZE = 16  # 增大按钮字体尺寸

# PIL图像转换为QImage的辅助函数
def pil_to_qimage(pil_image):
    """将PIL图像转换为QImage"""
    if pil_image.mode == "RGBA":
        qim = QImage(pil_image.tobytes("raw", "RGBA"), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    else:
        # 转换为RGB模式
        rgb_image = pil_image.convert("RGB")
        qim = QImage(rgb_image.tobytes("raw", "RGB"), rgb_image.width, rgb_image.height, QImage.Format_RGB888)
    return qim

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
        """加载图像"""
        try:
            logger.info(f"Loading image from: {image_path}")
            self.image = Image.open(image_path).convert("RGB")
            logger.info(f"Image loaded, size: {self.image.size}")
            self.update_display()
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            return False
    
    def update_display(self):
        """更新显示的图像"""
        if self.image is None:
            logger.warning("Cannot update display, image is None")
            return
        
        # 创建工作副本
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
        """更新QPixmap"""
        if self.display_image is None:
            logger.warning("Cannot update pixmap, display_image is None")
            return
        
        try:
            # 转换PIL图像为QPixmap
            qim = pil_to_qimage(self.display_image)
            
            # 根据控件大小缩放图像
            self.scaled_image = qim.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(self.scaled_image))
        except Exception as e:
            logger.error(f"Error updating pixmap: {str(e)}")
    
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
        """更新QPixmap"""
        if self.result_image is None:
            return
        
        try:
            # 转换PIL图像为QPixmap
            qim = pil_to_qimage(self.result_image)
            
            # 根据控件大小缩放图像
            self.scaled_image = qim.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(self.scaled_image))
        except Exception as e:
            logger.error(f"Error updating result pixmap: {str(e)}")
    
    def resizeEvent(self, event):
        """控件大小改变事件"""
        super().resizeEvent(event)
        self._update_pixmap()


class SAM2UI(QMainWindow):
    """SAM2交互式分割界面"""
    def __init__(self):
        super().__init__()
        self.predictor = None
        logger.info("Initializing SAM2UI")
        self.initUI()
    
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
                padding: 5px;
                font-weight: bold;
                margin-top: 20px;  /* 进一步减小顶部边距 */
                max-height: 100px;  /* 降低高度 */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                top: 20px;  /* 降低标题位置 */
                padding: 0 5px 0 5px;
                background-color: {COLORS['accent2']};
                color: #333333;
                font-size: {FONT_SIZE}px;
            }}
            /* 使单选按钮看起来像按钮 */
            QRadioButton {{
                font-size: {FONT_SIZE + 2}px;
                padding: 6px;
                spacing: 5px;
                min-height: 30px;
                background-color: {COLORS['background']};
                border-radius: 5px;
                margin: 2px;
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
        compact_options.setSpacing(20)  # 增加组之间的间距
        
        # 提示类型选项组 - 更紧凑的设计
        prompt_group = QGroupBox("提示类型")
        prompt_group.setStyleSheet(option_group_style)
        prompt_layout = QHBoxLayout()
        prompt_layout.setContentsMargins(5, 10, 5, 5)  # 减小内边距
        prompt_layout.setSpacing(20)  # 减小间距使按钮占据更多空间
        
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
        point_type_layout.setContentsMargins(5, 10, 5, 5)  # 减小内边距
        point_type_layout.setSpacing(20)  # 减小间距使按钮占据更多空间
        
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
            
            # 创建结果图像
            result_img = self.visualize_segmentation(self.image_canvas.image, best_mask)
            
            # 显示结果
            self.result_canvas.set_result(result_img)
            
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
        ui = SAM2UI()
        ui.show()
        logger.info("Application window shown")
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        print(f"Critical error: {str(e)}") 