import sys
import os
import numpy as np
import torch
import logging
import cv2

# 禁用PyTorch编译和Triton相关功能，避免错误
os.environ["TORCH_COMPILE_DISABLE_CUDA_GRAPHS"] = "1"
os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup,
                             QSlider, QGroupBox, QMessageBox, QSplitter, QProgressBar)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PIL import Image, ImageDraw

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义颜色常量 - 使用浅色配色方案
COLORS = {
    "background": "#FFFFFF",  # 系统UI界面底色，改为纯白色
    "foreground": "#F0F8FF",  # 组件颜色，改为爱丽丝蓝（非常浅的蓝色）
    "accent1": "#E1F5FE",     # 浅蓝色
    "accent2": "#E8F5E9",     # 浅绿色
    "text": "#333333",        # 深灰色文字，在浅色背景上提高可读性
    "success": "#C8E6C9",     # 浅绿色作为成功颜色
    "warning": "#FFCCBC",     # 浅橙色作为警告颜色
    "video_area": "#F5F5F5"   # 视频显示区域颜色，改为浅灰色
}

# 设置全局字体样式
FONT_FAMILY = "微软雅黑"  # 符合要求的字体
FONT_SIZE = 16           # 增大基础字体
BUTTON_FONT_SIZE = 18    # 增大按钮字体

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

# OpenCV图像转换为QImage的辅助函数
def cv2_to_qimage(cv_img):
    """将OpenCV图像转换为QImage"""
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    # OpenCV使用BGR顺序，需要转换为RGB
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

# OpenCV图像转换为PIL Image的辅助函数
def cv2_to_pil(cv_img):
    """将OpenCV图像转换为PIL Image"""
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img_rgb)

# 将NumPy掩码可视化为彩色叠加图像
def apply_mask_to_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """将分割掩码应用到图像上"""
    # 确保输入是NumPy数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 处理掩码类型
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 确保掩码是布尔类型或二值化的
    if not mask.dtype == bool:
        # 如果是浮点数，转换为二值掩码
        if np.issubdtype(mask.dtype, np.floating):
            mask = mask > 0.0
        # 如果是整数，转换为二值掩码
        elif np.issubdtype(mask.dtype, np.integer):
            mask = mask > 0
    
    # 确保掩码和图像尺寸匹配
    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape[-2:]  # 掩码可能有多个维度
    
    # 如果掩码形状和图像不匹配，需要调整大小
    if mask_h != image_h or mask_w != image_w:
        try:
            # 如果掩码是3D的 (batch, h, w)，取第一个
            if len(mask.shape) == 3:
                mask = mask[0]
            
            # 或者如果掩码是4D的 (batch, channels, h, w)，取第一个通道的第一个批次
            elif len(mask.shape) == 4:
                mask = mask[0, 0]
            
            # 调整掩码大小以匹配图像
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (image_w, image_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # 转回布尔型
            mask = mask_resized.astype(bool)
        except Exception as e:
            logger.error(f"Error resizing mask: {str(e)}")
            # 如果调整大小失败，创建一个空掩码
            mask = np.zeros((image_h, image_w), dtype=bool)
    
    # 创建彩色掩码
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    
    # 将掩码叠加到图像上
    return cv2.addWeighted(image, 1, color_mask, alpha, 0)

class VideoCanvas(QLabel):
    """用于显示视频帧和处理交互的画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            background-color: {COLORS['video_area']};
            border-radius: 12px;
            border: 2px solid #FFFFFF;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        """)
        
        # 视频和帧相关
        self.video_path = None
        self.current_frame = None
        self.display_frame = None
        self.scaled_frame = None
        self.frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.frames_cache = {}  # 缓存帧，避免重复读取
        
        # 交互状态
        self.drawing = False
        self.point_mode = True  # True: 点模式, False: 框模式
        self.foreground_point = True  # True: 前景点, False: 背景点
        
        # 点和框数据
        self.points = []  # 格式: [(x, y, is_foreground), ...]
        self.point_labels = []  # 1: 前景, 0: 背景
        self.box = None  # 格式: [x1, y1, x2, y2]
        self.current_box = None  # 当前正在绘制的框
        
        # 结果
        self.segmentation_results = {}  # 存储分割结果: {frame_idx: masks}
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        logger.info("VideoCanvas initialized")
    
    def load_video(self, video_path):
        """加载视频"""
        try:
            logger.info(f"Loading video from: {video_path}")
            
            # 关闭之前的视频
            if self.cap is not None:
                self.cap.release()
            
            # 打开新视频
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise Exception("无法打开视频文件")
            
            self.video_path = video_path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_index = 0
            self.frames_cache = {}
            self.segmentation_results = {}
            
            # 读取第一帧
            success = self.set_frame(0)
            
            if not success:
                raise Exception("无法读取视频帧")
                
            logger.info(f"Video loaded, total frames: {self.total_frames}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法加载视频: {str(e)}")
            return False
    
    def set_frame(self, frame_idx):
        """设置当前帧"""
        if self.cap is None:
            return False
        
        # 检查帧索引是否有效
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return False
        
        # 检查是否已缓存该帧
        if frame_idx in self.frames_cache:
            self.current_frame = self.frames_cache[frame_idx]
            self.frame_index = frame_idx
            self.update_display()
            return True
        
        # 设置帧位置并读取
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            logger.error(f"Failed to read frame {frame_idx}")
            return False
        
        # 缓存帧
        self.frames_cache[frame_idx] = frame
        self.current_frame = frame
        self.frame_index = frame_idx
        self.update_display()
        return True
    
    def next_frame(self):
        """显示下一帧"""
        return self.set_frame(self.frame_index + 1)
    
    def prev_frame(self):
        """显示上一帧"""
        return self.set_frame(self.frame_index - 1)
    
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
    
    def update_display(self):
        """更新显示的帧"""
        if self.current_frame is None:
            logger.warning("Cannot update display, current_frame is None")
            return
        
        # 创建工作副本
        self.display_frame = self.current_frame.copy()
        
        # 绘制点
        if self.points:
            for x, y, is_foreground in self.points:
                color = (0, 255, 0) if is_foreground else (0, 0, 255)  # 绿色为前景，红色为背景
                cv2.circle(self.display_frame, (x, y), 8, color, -1)
        
        # 绘制框
        if self.box:
            x0, y0, x1, y1 = self.box
            cv2.rectangle(self.display_frame, (x0, y0), (x1, y1), (255, 255, 0), 3)
        
        # 绘制当前正在绘制的框
        if self.current_box:
            x0, y0, x1, y1 = self.current_box
            cv2.rectangle(self.display_frame, (x0, y0), (x1, y1), (255, 255, 0), 3)
        
        # 叠加分割结果（如果有）
        if self.frame_index in self.segmentation_results:
            masks = self.segmentation_results[self.frame_index]
            for i, mask in enumerate(masks):
                # 根据对象索引选择不同的颜色
                color_idx = i % 5
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                color = colors[color_idx]
                
                # 应用掩码
                self.display_frame = apply_mask_to_image(
                    self.display_frame, mask, color=color, alpha=0.5
                )
        
        # 转换为QPixmap并显示
        self._update_pixmap()
    
    def _update_pixmap(self):
        """更新QPixmap"""
        if self.display_frame is None:
            logger.warning("Cannot update pixmap, display_frame is None")
            return
        
        try:
            # 转换OpenCV图像为QImage
            qim = cv2_to_qimage(self.display_frame)
            
            # 根据控件大小缩放图像
            self.scaled_frame = qim.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(self.scaled_frame))
        except Exception as e:
            logger.error(f"Error updating pixmap: {str(e)}")
    
    def get_frame_coordinates(self, pos):
        """将控件坐标转换为帧坐标"""
        if self.current_frame is None or self.scaled_frame is None:
            return None
        
        # 获取图像在控件中的位置
        img_rect = QRect(
            (self.width() - self.scaled_frame.width()) // 2,
            (self.height() - self.scaled_frame.height()) // 2,
            self.scaled_frame.width(),
            self.scaled_frame.height()
        )
        
        # 检查点击是否在图像内
        if not img_rect.contains(pos):
            return None
        
        # 计算相对于图像的坐标
        frame_height, frame_width = self.current_frame.shape[:2]
        x_ratio = frame_width / self.scaled_frame.width()
        y_ratio = frame_height / self.scaled_frame.height()
        
        x = int((pos.x() - img_rect.left()) * x_ratio)
        y = int((pos.y() - img_rect.top()) * y_ratio)
        
        return (x, y)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.current_frame is None:
            return
        
        pos = self.get_frame_coordinates(event.pos())
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
        if self.current_frame is None or not self.drawing or self.point_mode:
            return
        
        pos = self.get_frame_coordinates(event.pos())
        if pos is None:
            return
        
        # 更新当前框的终点
        self.current_box[2] = pos[0]
        self.current_box[3] = pos[1]
        self.update_display()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.current_frame is None or not self.drawing or self.point_mode:
            return
        
        pos = self.get_frame_coordinates(event.pos())
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
    
    def add_segmentation_result(self, frame_idx, masks):
        """添加分割结果"""
        # 确保掩码是NumPy数组
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        
        # 处理掩码维度
        # 确保掩码格式正确 - 应该是(num_objects, H, W)
        if len(masks.shape) == 4:  # (batch, channels, H, W)
            masks = masks[:, 0]    # 取第一个通道
        
        # 确保掩码是布尔类型或二值化的
        if not masks.dtype == bool:
            # 如果是浮点数，转换为二值掩码
            if np.issubdtype(masks.dtype, np.floating):
                masks = masks > 0.0
            # 如果是整数，转换为二值掩码
            elif np.issubdtype(masks.dtype, np.integer):
                masks = masks > 0
        
        # 确保掩码和图像尺寸匹配
        if self.current_frame is not None:
            image_h, image_w = self.current_frame.shape[:2]
            
            # 创建一个列表来存储调整大小后的掩码
            resized_masks = []
            
            # 处理每个对象的掩码
            for i in range(masks.shape[0]):
                mask = masks[i]
                
                # 如果掩码尺寸与图像不匹配，调整大小
                if mask.shape[0] != image_h or mask.shape[1] != image_w:
                    try:
                        # 调整掩码大小以匹配图像
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8), 
                            (image_w, image_h), 
                            interpolation=cv2.INTER_NEAREST
                        )
                        # 转回布尔型
                        resized_masks.append(mask_resized.astype(bool))
                    except Exception as e:
                        logger.error(f"Error resizing mask for object {i}: {str(e)}")
                        # 如果调整大小失败，创建一个空掩码
                        resized_masks.append(np.zeros((image_h, image_w), dtype=bool))
                else:
                    resized_masks.append(mask)
            
            # 更新掩码
            masks = np.array(resized_masks)
        
        self.segmentation_results[frame_idx] = masks
        if frame_idx == self.frame_index:
            self.update_display()
    
    def clear_segmentation_results(self):
        """清除所有分割结果"""
        self.segmentation_results = {}
        self.update_display()

class SAM2VideoUI(QMainWindow):
    """SAM2视频分割界面"""
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.inference_state = None
        self.is_tracking = False
        self.current_obj_id = 1  # 当前对象ID
        
        # 播放控制
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.play_speed = 51  # 默认约50ms，与滑块初始值匹配
        
        logger.info("Initializing SAM2VideoUI")
        self.initUI()
    
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('SAM2 视频分割')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QGroupBox {{
                font-weight: bold;
                border-radius: 10px;
                border: 2px solid #FFFFFF;
                margin-top: 12px;
                padding-top: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            QLabel {{
                color: {COLORS['text']};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE}px;
            }}
        """)
        
        # 设置应用字体
        app_font = QFont(FONT_FAMILY, FONT_SIZE)
        app_font.setStyleStrategy(QFont.PreferAntialias)
        QApplication.setFont(app_font)
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)  # 增加布局间距
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 统一按钮样式
        self.button_style = f"""
            QPushButton {{
                background-color: {COLORS['accent2']};
                color: #333333;
                border: none;
                border-radius: 10px;
                padding: 12px 20px;
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #FFFFFF;
                color: {COLORS['background']};
            }}
            QPushButton:pressed {{
                background-color: #D0EFD0;
                border-color: #D0EFD0;
            }}
            QPushButton:disabled {{
                background-color: #A0D0E0;
                color: #666666;
                border-color: #A0D0E0;
            }}
        """
        
        # 创建顶部工具栏
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(20, 20, 20, 20)  # 增加边距
        toolbar_layout.setSpacing(20)  # 增加间距
        
        # 加载视频按钮
        load_video_btn = QPushButton("加载视频")
        load_video_btn.clicked.connect(self.load_video)
        load_video_btn.setStyleSheet(self.button_style)
        toolbar_layout.addWidget(load_video_btn)
        
        # 加载模型按钮
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        load_model_btn.setStyleSheet(self.button_style)
        toolbar_layout.addWidget(load_model_btn)
        
        # 清除提示按钮
        clear_btn = QPushButton("清除提示")
        clear_btn.clicked.connect(self.clear_prompts)
        clear_btn.setStyleSheet(self.button_style)
        toolbar_layout.addWidget(clear_btn)
        
        # 添加分割按钮
        segment_btn = QPushButton("添加分割")
        segment_btn.clicked.connect(self.add_segmentation)
        segment_btn.setStyleSheet(self.button_style)
        toolbar_layout.addWidget(segment_btn)
        
        # 追踪按钮 - 使用强调色
        track_btn = QPushButton("开始追踪")
        track_btn.clicked.connect(self.track_objects)
        track_btn.setStyleSheet(self.button_style + f"""
            background-color: #FFFFFF;
            color: {COLORS['background']};
            font-weight: bold;
        """)
        toolbar_layout.addWidget(track_btn)
        
        # 保存结果按钮
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self.save_result)
        save_btn.setStyleSheet(self.button_style)
        toolbar_layout.addWidget(save_btn)
        
        main_layout.addWidget(toolbar_widget)
        
        # 创建选项组
        options_widget = QWidget()
        options_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
                border-radius: 10px;
            }}
        """)
        options_layout = QHBoxLayout(options_widget)
        options_layout.setContentsMargins(20, 15, 20, 15)  # 增加边距
        
        # 选项组样式
        option_group_style = f"""
            QGroupBox {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
                margin-top: 12px;
                border: 2px solid #FFFFFF;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                top: 0px;
                padding: 0 8px;
                background-color: {COLORS['foreground']};
                color: {COLORS['text']};
                font-size: {FONT_SIZE + 2}px;
            }}
            QRadioButton {{
                font-size: {FONT_SIZE}px;
                padding: 8px;
                spacing: 8px;
                background-color: {COLORS['background']};
                border-radius: 8px;
                margin: 5px;
                color: #FFFFFF;
            }}
            QRadioButton:checked {{
                font-weight: bold;
                color: #333333;
                background-color: {COLORS['accent2']};
            }}
        """
        
        # 提示类型选项组
        compact_options = QHBoxLayout()
        compact_options.setSpacing(15)
        
        prompt_group = QGroupBox("提示类型")
        prompt_group.setStyleSheet(option_group_style)
        prompt_layout = QHBoxLayout()
        prompt_layout.setContentsMargins(10, 15, 10, 5)
        prompt_layout.setSpacing(10)
        
        self.point_radio = QRadioButton("点")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(self.toggle_prompt_type)
        
        self.box_radio = QRadioButton("框")
        
        prompt_layout.addWidget(self.point_radio, 1)
        prompt_layout.addWidget(self.box_radio, 1)
        prompt_group.setLayout(prompt_layout)
        
        # 点类型选项组
        point_type_group = QGroupBox("点类型")
        point_type_group.setStyleSheet(option_group_style)
        point_type_layout = QHBoxLayout()
        point_type_layout.setContentsMargins(10, 15, 10, 5)
        point_type_layout.setSpacing(10)
        
        self.foreground_radio = QRadioButton("前景")
        self.foreground_radio.setChecked(True)
        self.foreground_radio.toggled.connect(self.toggle_point_type)
        
        self.background_radio = QRadioButton("背景")
        
        point_type_layout.addWidget(self.foreground_radio, 1)
        point_type_layout.addWidget(self.background_radio, 1)
        point_type_group.setLayout(point_type_layout)
        
        # 对象ID选择组
        obj_id_group = QGroupBox("对象ID")
        obj_id_group.setStyleSheet(option_group_style)
        obj_id_layout = QHBoxLayout()
        obj_id_layout.setContentsMargins(10, 15, 10, 5)
        
        # 添加对象ID标签
        self.obj_id_label = QLabel("当前对象: 1")
        obj_id_layout.addWidget(self.obj_id_label)
        
        # 添加增减按钮
        dec_id_btn = QPushButton("-")
        dec_id_btn.clicked.connect(self.decrease_obj_id)
        dec_id_btn.setMaximumWidth(30)
        dec_id_btn.setStyleSheet(self.button_style)
        obj_id_layout.addWidget(dec_id_btn)
        
        inc_id_btn = QPushButton("+")
        inc_id_btn.clicked.connect(self.increase_obj_id)
        inc_id_btn.setMaximumWidth(30)
        inc_id_btn.setStyleSheet(self.button_style)
        obj_id_layout.addWidget(inc_id_btn)
        
        obj_id_group.setLayout(obj_id_layout)
        
        # 将选项组添加到布局
        compact_options.addWidget(prompt_group, 1)
        compact_options.addWidget(point_type_group, 1)
        compact_options.addWidget(obj_id_group, 1)
        
        options_layout.addLayout(compact_options)
        main_layout.addWidget(options_widget)
        
        # 创建视频播放控制
        playback_widget = QWidget()
        playback_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
                border: 2px solid #FFFFFF;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            QSlider::groove:horizontal {{
                height: 10px;
                background: {COLORS['background']};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent2']};
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent2']};
                border-radius: 5px;
            }}
        """)
        playback_layout = QHBoxLayout(playback_widget)
        playback_layout.setContentsMargins(20, 15, 20, 15)  # 增加边距
        
        # 帧计数和控制
        self.frame_label = QLabel("帧: 0 / 0")
        self.frame_label.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        playback_layout.addWidget(self.frame_label)
        
        # 添加播放/暂停按钮
        self.play_pause_btn = QPushButton("播放")
        self.play_pause_btn.clicked.connect(self.toggle_play)
        self.play_pause_btn.setStyleSheet(self.button_style)
        playback_layout.addWidget(self.play_pause_btn)
        
        # 添加上一帧按钮
        prev_frame_btn = QPushButton("◀")
        prev_frame_btn.clicked.connect(self.prev_frame)
        prev_frame_btn.setMaximumWidth(50)
        prev_frame_btn.setStyleSheet(self.button_style)
        playback_layout.addWidget(prev_frame_btn)
        
        # 添加下一帧按钮
        next_frame_btn = QPushButton("▶")
        next_frame_btn.clicked.connect(self.next_frame)
        next_frame_btn.setMaximumWidth(50)
        next_frame_btn.setStyleSheet(self.button_style)
        playback_layout.addWidget(next_frame_btn)
        
        # 添加播放速度控制
        speed_label = QLabel("速度:")
        speed_label.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        playback_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)  # 最慢 (100ms)
        self.speed_slider.setMaximum(100)  # 最快 (2ms)
        self.speed_slider.setValue(50)  # 默认50 (约51ms)
        self.speed_slider.setMaximumWidth(150)
        self.speed_slider.valueChanged.connect(self.change_playback_speed)
        playback_layout.addWidget(self.speed_slider)

        # 添加帧滑块
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        playback_layout.addWidget(self.frame_slider, 3)  # 给滑块更多空间
        
        main_layout.addWidget(playback_widget)
        
        # 创建视频显示区域
        self.video_canvas = VideoCanvas()
        main_layout.addWidget(self.video_canvas, 1)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #FFFFFF;
                border-radius: 10px;
                text-align: center;
                background-color: {COLORS['background']};
                height: 25px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE}px;
                color: #FFFFFF;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent2']};
                width: 15px;
                margin: 0px;
                border-radius: 5px;
            }}
        """)
        main_layout.addWidget(self.progress_bar)
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        self.statusBar().setStyleSheet(f"""
            background-color: {COLORS['foreground']};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE}px;
            padding: 8px;
            color: #FFFFFF;
            border-top: 2px solid #FFFFFF;
        """)
        
        logger.info("UI initialization complete")
    
    def load_video(self):
        """加载视频"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)", options=options
        )
        
        if file_path:
            logger.info(f"Selected video file: {file_path}")
            
            # 暂停播放
            self.stop_playback()
            
            # 加载视频
            if self.video_canvas.load_video(file_path):
                self.statusBar().showMessage(f'已加载视频: {os.path.basename(file_path)}')
                
                # 更新帧滑块和标签
                self.frame_slider.setMaximum(self.video_canvas.total_frames - 1)
                self.frame_slider.setValue(0)
                self.update_frame_label()
                
                # 重置分割状态
                self.inference_state = None
                self.is_tracking = False
                self.current_obj_id = 1
                self.update_obj_id_label()
    
    def load_model(self):
        """加载SAM2模型"""
        try:
            # 显示模型选择对话框
            model_size = self.select_model_size()
            if not model_size:
                self.statusBar().showMessage('模型加载取消')
                return
                
            self.statusBar().showMessage('正在加载模型...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            QApplication.processEvents()  # 更新UI
            
            # 检查是否选择了CPU模式
            use_cpu = False
            if model_size.endswith("_cpu"):
                use_cpu = True
                model_size = model_size.replace("_cpu", "")
                logger.info(f"Loading SAM2 model: {model_size} in CPU mode")
                self.statusBar().showMessage(f'正在使用CPU模式加载模型 {model_size}...')
            else:
                logger.info(f"Loading SAM2 model: {model_size}")
            
            # 根据选择的大小确定模型ID
            model_id = f"facebook/sam2.1-hiera-{model_size}"
            
            # 显示正在下载/加载的提示
            self.statusBar().showMessage(f'正在下载/加载模型: {model_id}...')
            self.progress_bar.setValue(30)
            QApplication.processEvents()  # 更新UI
            
            # 尝试加载模型
            try:
                self.load_sam2_model(model_id, use_cpu)
                
                self.progress_bar.setValue(100)
                device_type = "CPU" if use_cpu else "GPU"
                self.statusBar().showMessage(f'模型 {model_size} 在{device_type}上加载成功')
                self.progress_bar.setVisible(False)
                
                # 显示成功消息
                QMessageBox.information(self, "成功", f"SAM2 模型 ({model_size}) 已成功在{device_type}上加载！\n现在可以加载视频并开始分割。")
                
            except RuntimeError as e:
                # 处理CUDA内存不足错误
                if "CUDA out of memory" in str(e) or "device-side assert" in str(e):
                    self.handle_cuda_memory_error(model_size)
                else:
                    # 其他RuntimeError
                    raise e
            except Exception as e:
                if "404" in str(e):
                    logger.error(f"Model not found on Hugging Face: {str(e)}")
                    QMessageBox.critical(self, "错误", f"在Hugging Face上找不到模型 {model_id}，请检查网络连接或选择其他模型")
                else:
                    logger.error(f"Error downloading model: {str(e)}")
                    QMessageBox.critical(self, "错误", f"下载模型时出错: {str(e)}")
                self.statusBar().showMessage('模型加载失败')
                self.progress_bar.setVisible(False)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"无法加载模型: {str(e)}")
            self.statusBar().showMessage('模型加载失败')
            self.progress_bar.setVisible(False)
    
    def load_sam2_model(self, model_id, use_cpu=False):
        """加载SAM2模型的实际函数，包含错误处理"""
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            
            # 设置进度更新
            self.statusBar().showMessage(f'正在下载模型文件...')
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            # 加载模型
            device = "cpu" if use_cpu else "cuda"
            
            # 设置一些环境变量，避免可能的错误
            os.environ["TORCH_COMPILE_DISABLE_CUDA_GRAPHS"] = "1"
            os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
            
            # 尝试加载模型，禁用torch.compile避免错误
            self.predictor = SAM2VideoPredictor.from_pretrained(
                model_id, 
                vos_optimized=False,  # 禁用torch.compile
                device=device
            )
            
            self.progress_bar.setValue(90)
            self.statusBar().showMessage(f'模型加载完成，正在初始化...')
            QApplication.processEvents()
            
            return True
            
        except ImportError as e:
            logger.error(f"SAM2 module not found: {str(e)}")
            QMessageBox.critical(self, "错误", "无法导入SAM2模块，请确保正确安装了SAM2")
            self.statusBar().showMessage('模型加载失败: 模块未找到')
            self.progress_bar.setVisible(False)
            raise
    
    def handle_cuda_memory_error(self, current_model_size):
        """处理CUDA内存不足错误的专用函数"""
        logger.error(f"CUDA out of memory when loading {current_model_size} model")
        
        # 提供选项
        retry = QMessageBox.question(
            self, 
            "GPU内存不足", 
            f"您的GPU内存不足以加载{current_model_size}模型。\n\n是否尝试使用CPU模式加载tiny模型？\n(注意：CPU模式会非常慢)",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if retry == QMessageBox.Yes:
            # 清理之前的尝试
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 设置为使用CPU
            self.statusBar().showMessage('正在使用CPU模式重试...')
            QApplication.processEvents()
            
            # 使用CPU模式加载tiny模型
            try:
                model_id = "facebook/sam2.1-hiera-tiny"
                self.load_sam2_model(model_id, use_cpu=True)
                
                self.progress_bar.setValue(100)
                logger.info("SAM2 model loaded successfully on CPU")
                self.statusBar().showMessage(f'模型已在CPU上加载成功')
                self.progress_bar.setVisible(False)
                
                # 显示成功消息
                QMessageBox.information(self, "成功", f"SAM2 tiny模型已在CPU上成功加载！\n注意：CPU模式下处理速度会较慢。")
                
            except Exception as cpu_error:
                logger.error(f"Failed to load model on CPU: {str(cpu_error)}")
                QMessageBox.critical(self, "错误", f"在CPU上加载模型失败: {str(cpu_error)}")
                self.statusBar().showMessage('模型加载失败')
                self.progress_bar.setVisible(False)
                raise cpu_error
        else:
            self.statusBar().showMessage('模型加载取消')
            self.progress_bar.setVisible(False)
    
    def select_model_size(self):
        """选择模型大小"""
        sizes = {
            "tiny": "最小 (适合低配置设备)",
            "small": "小型 (适合中等配置)",
            "base_plus": "中型+ (推荐配置)",
            "large": "大型 (高端配置)"
        }
        
        # 创建选择对话框
        dialog = QMessageBox(self)
        dialog.setWindowTitle("选择模型大小")
        dialog.setText("请选择要加载的SAM2模型大小：\n(根据您的设备性能选择合适的大小)")
        dialog.setIcon(QMessageBox.Question)
        
        # 添加按钮
        buttons = {}
        for size, description in sizes.items():
            button = dialog.addButton(f"{size} - {description}", QMessageBox.ActionRole)
            buttons[button] = size
        
        # 添加CPU模式选项
        cpu_button = dialog.addButton("CPU模式 (使用tiny模型)", QMessageBox.ActionRole)
        buttons[cpu_button] = "tiny_cpu"
        
        cancel_button = dialog.addButton("取消", QMessageBox.RejectRole)
        
        # 显示对话框并获取结果
        dialog.exec_()
        
        clicked_button = dialog.clickedButton()
        if clicked_button == cancel_button:
            return None
        
        return buttons[clicked_button]
    
    def toggle_prompt_type(self):
        """切换提示类型"""
        is_point_mode = self.point_radio.isChecked()
        self.video_canvas.set_point_mode(is_point_mode)
    
    def toggle_point_type(self):
        """切换点类型"""
        is_foreground = self.foreground_radio.isChecked()
        self.video_canvas.set_foreground_point(is_foreground)
    
    def clear_prompts(self):
        """清除所有提示"""
        self.video_canvas.clear_prompts()
        self.statusBar().showMessage('已清除所有提示')
    
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
        self.obj_id_label.setText(f"当前对象: {self.current_obj_id}")
    
    def add_segmentation(self):
        """为当前帧添加分割"""
        if self.predictor is None:
            logger.warning("Segmentation attempted without loading model")
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if self.video_canvas.current_frame is None:
            logger.warning("Segmentation attempted without loading video")
            QMessageBox.warning(self, "警告", "请先加载视频")
            return
        
        # 获取提示数据
        point_coords, point_labels, box = self.video_canvas.get_prompt_data()
        
        if point_coords is None and box is None:
            logger.warning("Segmentation attempted without prompts")
            QMessageBox.warning(self, "警告", "请添加至少一个点或一个框")
            return
        
        try:
            # 首次分割需要初始化
            if self.inference_state is None:
                logger.info("Initializing inference state...")
                self.statusBar().showMessage('初始化分割状态...')
                
                try:
                    with torch.inference_mode():
                        # 检查是否在CPU上运行
                        if self.predictor.device.type == "cpu":
                            self.statusBar().showMessage('在CPU上初始化中，可能需要较长时间...')
                            QApplication.processEvents()
                            
                        self.inference_state = self.predictor.init_state(self.video_canvas.video_path)
                        
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"CUDA out of memory during initialization: {str(e)}")
                        QMessageBox.critical(self, "错误", "GPU内存不足，无法初始化视频状态。请尝试使用CPU模式或更小的模型。")
                        self.statusBar().showMessage('初始化失败：GPU内存不足')
                        return
                    else:
                        raise e
                    
                logger.info("Inference state initialized")
            
            # 添加新的点或框
            self.statusBar().showMessage('正在分割...')
            logger.info(f"Adding segmentation for object {self.current_obj_id} on frame {self.video_canvas.frame_index}")
            
            try:
                with torch.inference_mode():
                    # 如果在CPU上运行，提示用户可能需要等待较长时间
                    if self.predictor.device.type == "cpu":
                        self.statusBar().showMessage('在CPU上处理中，请耐心等待...')
                        QApplication.processEvents()
                    
                    # 使用自动类型转换，避免CUDA/CPU类型不匹配错误
                    if point_coords is not None:
                        point_coords = point_coords.astype(np.float32)
                        logger.info(f"Point coords shape: {point_coords.shape}")
                    if point_labels is not None:
                        logger.info(f"Point labels shape: {point_labels.shape}")
                    if box is not None:
                        box = box.astype(np.float32)
                        logger.info(f"Box shape: {box.shape}, values: {box}")
                    
                    # 打印当前视频分辨率以进行调试
                    video_h, video_w = self.video_canvas.current_frame.shape[:2]
                    logger.info(f"Video frame size: {video_w}x{video_h}")
                    
                    # 捕获特定错误并提供详细信息
                    try:
                        frame_idx, obj_ids, masks = self.predictor.add_new_points_or_box(
                            self.inference_state,
                            frame_idx=self.video_canvas.frame_index,
                            obj_id=self.current_obj_id,
                            points=point_coords,
                            labels=point_labels,
                            box=box,
                            normalize_coords=True
                        )
                        
                        # 打印掩码信息
                        if isinstance(masks, torch.Tensor):
                            logger.info(f"Mask tensor shape: {masks.shape}, device: {masks.device}")
                            masks = masks.cpu().numpy()
                        logger.info(f"Masks numpy shape: {masks.shape}")
                        
                    except ValueError as ve:
                        logger.error(f"ValueError in add_new_points_or_box: {str(ve)}")
                        if "boolean index did not match" in str(ve):
                            # 创建一个空掩码作为替代
                            logger.info("Creating empty mask as fallback due to shape mismatch")
                            masks = np.zeros((1, video_h, video_w), dtype=bool)
                            frame_idx = self.video_canvas.frame_index
                            obj_ids = [self.current_obj_id]
                        else:
                            raise
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA out of memory during segmentation: {str(e)}")
                    QMessageBox.critical(self, "错误", "GPU内存不足，无法完成分割。请尝试使用CPU模式或更小的模型。")
                    self.statusBar().showMessage('分割失败：GPU内存不足')
                    return
                else:
                    raise e
            
            # 将结果添加到视频画布
            logger.info(f"Adding segmentation result for frame {frame_idx}, mask shape: {masks.shape}")
            self.video_canvas.add_segmentation_result(frame_idx, masks)
            
            self.statusBar().showMessage(f'已添加分割，对象ID: {self.current_obj_id}')
            logger.info(f"Segmentation added for object {self.current_obj_id}")
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")
            self.statusBar().showMessage('分割失败')
    
    def track_objects(self):
        """追踪视频中的对象"""
        if self.predictor is None:
            logger.warning("Tracking attempted without loading model")
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if self.inference_state is None:
            logger.warning("Tracking attempted without segmentation")
            QMessageBox.warning(self, "警告", "请先添加至少一个分割")
            return
        
        if self.is_tracking:
            logger.warning("Tracking already in progress")
            QMessageBox.information(self, "提示", "追踪已在进行中")
            return
        
        try:
            self.is_tracking = True
            self.statusBar().showMessage('正在追踪对象...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 暂停播放
            self.stop_playback()
            
            # 追踪对象
            logger.info("Starting object tracking")
            
            # 检查是否在CPU上运行
            is_cpu_mode = self.predictor.device.type == "cpu"
            if is_cpu_mode:
                # 在CPU模式下提示用户可能需要很长时间
                warning = QMessageBox.warning(
                    self, 
                    "CPU模式警告", 
                    "您正在使用CPU模式，追踪过程可能非常慢。\n\n是否继续？",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if warning == QMessageBox.No:
                    self.is_tracking = False
                    self.statusBar().showMessage('追踪已取消')
                    self.progress_bar.setVisible(False)
                    return
                
                self.statusBar().showMessage('在CPU上追踪中，请耐心等待...')
            
            # 使用Torch的推理模式
            try:
                with torch.inference_mode():
                    # 追踪生成器
                    tracking_generator = self.predictor.propagate_in_video(
                        self.inference_state,
                        start_frame_idx=self.video_canvas.frame_index
                    )
                    
                    # 获取总帧数以更新进度条
                    total_frames = self.video_canvas.total_frames
                    
                    # 处理每一帧
                    for frame_idx, obj_ids, masks in tracking_generator:
                        # 更新进度条
                        progress = int((frame_idx / total_frames) * 100)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()  # 更新UI
                        
                        # 确保掩码在CPU上，以便可以转换为NumPy数组
                        try:
                            # 打印掩码信息以进行调试
                            if isinstance(masks, torch.Tensor):
                                logger.info(f"Tracking frame {frame_idx}, mask tensor shape: {masks.shape}, device: {masks.device}")
                                # 安全地将掩码转换到CPU
                                if masks.device.type != "cpu":
                                    masks = masks.detach().cpu()
                                masks = masks.numpy()
                            
                            logger.info(f"Mask numpy shape: {masks.shape}")
                            
                            # 添加分割结果
                            self.video_canvas.add_segmentation_result(frame_idx, masks)
                            
                            # 日志
                            logger.info(f"Tracked frame {frame_idx}, objects: {obj_ids}")
                        except Exception as mask_err:
                            logger.error(f"Error processing masks for frame {frame_idx}: {str(mask_err)}")
                            # 创建一个空掩码作为替代
                            video_h, video_w = self.video_canvas.current_frame.shape[:2]
                            empty_masks = np.zeros((len(obj_ids), video_h, video_w), dtype=bool)
                            self.video_canvas.add_segmentation_result(frame_idx, empty_masks)
                        
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA out of memory during tracking: {str(e)}")
                    QMessageBox.critical(self, "错误", "GPU内存不足，无法完成追踪。请尝试使用CPU模式或更小的模型。")
                    self.statusBar().showMessage('追踪失败：GPU内存不足')
                    self.progress_bar.setVisible(False)
                    self.is_tracking = False
                    return
                else:
                    raise e
            
            self.progress_bar.setValue(100)
            self.is_tracking = False
            self.statusBar().showMessage('追踪完成')
            logger.info("Object tracking completed")
            
            # 提示用户追踪完成
            QMessageBox.information(self, "提示", "对象追踪完成，可以使用播放按钮查看结果")
            
            # 隐藏进度条
            self.progress_bar.setVisible(False)
        except Exception as e:
            self.is_tracking = False
            logger.error(f"Tracking error: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"追踪失败: {str(e)}")
            self.statusBar().showMessage('追踪失败')
            self.progress_bar.setVisible(False)
    
    def save_result(self):
        """保存分割结果视频"""
        if not self.video_canvas.segmentation_results:
            logger.warning("Save result attempted without segmentation results")
            QMessageBox.warning(self, "警告", "没有可保存的分割结果")
            return
        
        try:
            # 选择保存路径
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果视频", "", "MP4视频 (*.mp4);;所有文件 (*)", options=options
            )
            
            if not file_path:
                return
            
            # 确保文件扩展名正确
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            
            self.statusBar().showMessage('正在保存视频...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # 更新UI
            
            # 获取视频参数
            height, width = self.video_canvas.current_frame.shape[:2]
            fps = 30  # 假设30fps
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # 遍历所有帧并保存
            for frame_idx in range(self.video_canvas.total_frames):
                # 更新进度条
                progress = int((frame_idx / self.video_canvas.total_frames) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()  # 更新UI
                
                # 设置当前帧
                self.video_canvas.set_frame(frame_idx)
                
                # 写入帧
                out.write(self.video_canvas.display_frame)
            
            # 释放写入器
            out.release()
            
            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f'视频已保存至: {file_path}')
            self.progress_bar.setVisible(False)
            
            QMessageBox.information(self, "提示", f"视频已成功保存至:\n{file_path}")
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            self.statusBar().showMessage('保存失败')
            self.progress_bar.setVisible(False)
    
    def update_frame_label(self):
        """更新帧标签"""
        if self.video_canvas.cap is not None:
            self.frame_label.setText(f"帧: {self.video_canvas.frame_index + 1} / {self.video_canvas.total_frames}")
    
    def slider_frame_changed(self, value):
        """滑块帧改变"""
        if self.video_canvas.cap is not None:
            self.video_canvas.set_frame(value)
            self.update_frame_label()
    
    def next_frame(self):
        """显示下一帧"""
        if self.video_canvas.cap is not None:
            if self.video_canvas.next_frame():
                self.frame_slider.setValue(self.video_canvas.frame_index)
                self.update_frame_label()
    
    def prev_frame(self):
        """显示上一帧"""
        if self.video_canvas.cap is not None:
            if self.video_canvas.prev_frame():
                self.frame_slider.setValue(self.video_canvas.frame_index)
                self.update_frame_label()
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.play_timer.isActive():
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """开始播放"""
        self.play_timer.start(self.play_speed)
        self.play_pause_btn.setText("暂停")
    
    def stop_playback(self):
        """停止播放"""
        self.play_timer.stop()
        self.play_pause_btn.setText("播放")
    
    def play_next_frame(self):
        """播放下一帧"""
        # 如果到达视频末尾，停止播放
        if self.video_canvas.frame_index >= self.video_canvas.total_frames - 1:
            self.stop_playback()
            return
        
        self.next_frame()
    
    def change_playback_speed(self):
        """改变播放速度"""
        # 将滑块值(1-100)转换为播放间隔(2-100ms)，使用非线性映射以获得更好的控制感
        value = self.speed_slider.value()
        # 使用指数映射，值越大，播放间隔越小（速度越快）
        # 2ms (最快) 到 100ms (最慢)
        self.play_speed = int(100 - (value * 0.98) + 2)
        
        logger.info(f"Playback speed changed: {self.play_speed}ms interval")
        
        # 如果正在播放，重新启动计时器以应用新的间隔
        if self.play_timer.isActive():
            self.play_timer.start(self.play_speed)


if __name__ == "__main__":
    try:
        logger.info("Starting SAM2 Video UI application")
        app = QApplication(sys.argv)
        ui = SAM2VideoUI()
        ui.show()
        logger.info("Application window shown")
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        print(f"Critical error: {str(e)}") 