import sys
import os
import numpy as np
import torch
import logging
import cv2
import requests
import time
import urllib3

# 禁用PyTorch编译和Triton相关功能，避免错误
os.environ["TORCH_COMPILE_DISABLE_CUDA_GRAPHS"] = "1"
os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup,
                             QSlider, QGroupBox, QMessageBox, QSplitter, QProgressBar, QProgressDialog,
                             QDialog, QDialogButtonBox, QCheckBox)
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

# 定义模型信息 - 添加在颜色常量后面
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
    
    # 在sam2目录下创建models文件夹
    model_dir = os.path.join(script_dir, "models")
    
    # 如果目录不存在，创建它
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Using models directory: {model_dir}")
    return model_dir

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

def download_model(model_info, save_path, parent_widget=None):
    """下载模型文件，带进度条和重试机制"""
    # 创建进度对话框
    from PyQt5.QtWidgets import QProgressDialog, QMessageBox
    from PyQt5.QtCore import Qt
    
    progress = QProgressDialog("正在下载模型...", "取消", 0, 100, parent_widget)
    progress.setWindowTitle("下载模型")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.setStyleSheet(f"""
        QProgressDialog {{
            background-color: {COLORS['foreground']};
            border-radius: 10px;
            min-width: 400px;
        }}
        QLabel {{
            color: {COLORS['text']};
            font-size: 14px;
            margin-bottom: 10px;
        }}
        QPushButton {{
            background-color: {COLORS['accent2']};
            color: {COLORS['text']};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
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
                        if parent_widget:
                            parent_widget.statusBar().showMessage('取消下载')
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
                    parent_widget, "下载错误", retry_msg, 
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
                    parent_widget, 
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
                backup_options = QMessageBox(parent_widget)
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
                        parent_widget,
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
        self.semantic_mode = False  # 新增：语义分割模式
        self.foreground_point = True  # True: 前景点, False: 背景点
        
        # 点和框数据
        self.points = []  # 格式: [(x, y, is_foreground), ...]
        self.point_labels = []  # 1: 前景, 0: 背景
        self.box = None  # 格式: [x1, y1, x2, y2]
        self.current_box = None  # 当前正在绘制的框
        
        # 结果
        self.segmentation_results = {}  # 存储分割结果: {frame_idx: masks}
        
        # 缩放
        self.zoom_factor = 1.0
        self.max_zoom = 5.0
        self.min_zoom = 0.5
        
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
            
            # 应用缩放
            if self.zoom_factor != 1.0:
                orig_width = qim.width()
                orig_height = qim.height()
                
                # 计算缩放后的尺寸
                new_width = int(orig_width * self.zoom_factor)
                new_height = int(orig_height * self.zoom_factor)
                
                # 缩放图像
                qim = qim.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
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
    
    def set_semantic_mode(self, is_semantic_mode):
        """设置是否为语义分割模式"""
        self.semantic_mode = is_semantic_mode
        logger.info(f"Semantic mode set to: {is_semantic_mode}")
        # 语义分割模式下清除所有提示
        if is_semantic_mode:
            self.clear_prompts()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件处理，实现缩放功能"""
        if self.current_frame is None:
            return
        
        # 获取滚轮滚动的角度
        delta = event.angleDelta().y()
        
        # 根据滚轮方向调整缩放因子
        if delta > 0:
            # 向上滚，放大
            self.zoom_factor = min(self.max_zoom, self.zoom_factor * 1.1)
        else:
            # 向下滚，缩小
            self.zoom_factor = max(self.min_zoom, self.zoom_factor / 1.1)
        
        # 更新显示
        self.update_display()
        
        logger.info(f"Zoom factor changed to: {self.zoom_factor}")

class SAM2VideoUI(QMainWindow):
    """SAM2视频分割界面"""
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.inference_state = None
        self.is_tracking = False
        self.current_obj_id = 1  # 当前对象ID
        
        # 语义分割阈值
        self.semantic_threshold = 0.5  # 默认阈值
        
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
                background-color: #D0EFD0;
                border: 2px solid #FFFFFF;
            }}
            QPushButton:pressed {{
                background-color: #B0E0B0;
                border: 2px solid #FFFFFF;
            }}
            QPushButton:disabled {{
                background-color: #A0D0E0;
                color: #666666;
            }}
        """
        
        # 创建顶部工具栏
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
                border: 2px solid #FFFFFF;
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
        track_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #FFE0B0;
                color: #333333;
                border: 2px solid #FFFFFF;
                border-radius: 10px;
                padding: 12px 20px;
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #FFD090;
                border: 2px solid #FFFFFF;
            }}
            QPushButton:pressed {{
                background-color: #FFBB70;
                border: 2px solid #FFFFFF;
            }}
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
                color: {COLORS['text']};
                font-weight: normal;
            }}
            QRadioButton:checked {{
                font-weight: bold;
                color: #333333;
                background-color: {COLORS['accent2']};
                border: 2px solid #FFFFFF;
            }}
            QRadioButton:hover {{
                background-color: #D0EFD0;
                border: 1px solid #FFFFFF;
            }}
        """
        
        # 提示类型选项组
        compact_options = QHBoxLayout()
        compact_options.setSpacing(15)
        
        prompt_group = QGroupBox("提示类型")
        prompt_group.setStyleSheet(option_group_style)
        prompt_layout = QVBoxLayout()  # 改为垂直布局
        prompt_layout.setContentsMargins(10, 15, 10, 5)
        prompt_layout.setSpacing(10)
        
        # 第一行：点和框选项
        prompt_row1 = QHBoxLayout()
        prompt_row1.setSpacing(10)
        
        self.point_radio = QRadioButton("点标注")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(self.toggle_prompt_type)
        
        self.box_radio = QRadioButton("框标注")
        self.box_radio.toggled.connect(self.toggle_prompt_type)
        
        prompt_row1.addWidget(self.point_radio, 1)
        prompt_row1.addWidget(self.box_radio, 1)
        
        # 第二行：语义分割选项
        prompt_row2 = QHBoxLayout()
        prompt_row2.setSpacing(10)
        
        self.semantic_radio = QRadioButton("语义分割")
        self.semantic_radio.toggled.connect(self.toggle_prompt_type)
        
        prompt_row2.addWidget(self.semantic_radio, 1)
        prompt_row2.addStretch(1)
        
        # 添加两行到提示类型布局
        prompt_layout.addLayout(prompt_row1)
        prompt_layout.addLayout(prompt_row2)
        
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
        
        # 语义分割选项组 (默认隐藏)
        self.semantic_options_group = QGroupBox("语义分割选项")
        self.semantic_options_group.setStyleSheet(option_group_style)
        self.semantic_options_group.setVisible(False)  # 默认隐藏
        semantic_options_layout = QVBoxLayout()
        semantic_options_layout.setContentsMargins(10, 15, 10, 5)
        semantic_options_layout.setSpacing(10)
        
        # 语义分割说明
        semantic_desc = QLabel("自动对整个视频帧进行语义分割，识别所有物体")
        semantic_desc.setStyleSheet(f"""
            color: {COLORS['text']};
            font-size: 12px;
            padding: 5px;
            background-color: {COLORS['background']};
            border-radius: 4px;
            line-height: 1.4;
        """)
        semantic_desc.setWordWrap(True)
        semantic_options_layout.addWidget(semantic_desc)
        
        # 分割阈值
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(10)
        
        threshold_label = QLabel("分割阈值:")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)  # 默认值50%
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("50%")
        self.threshold_value_label.setStyleSheet(f"""
            color: {COLORS['accent2']};
            font-weight: bold;
            min-width: 40px;
            text-align: right;
        """)
        threshold_layout.addWidget(self.threshold_value_label)
        
        semantic_options_layout.addWidget(threshold_container)
        self.semantic_options_group.setLayout(semantic_options_layout)
        
        # 将选项组添加到布局
        compact_options.addWidget(prompt_group, 1)
        compact_options.addWidget(point_type_group, 1)
        compact_options.addWidget(obj_id_group, 1)
        compact_options.addWidget(self.semantic_options_group, 1)
        
        options_layout.addLayout(compact_options)
        main_layout.addWidget(options_widget)
        
        # 创建视频播放控制
        playback_widget = QWidget()
        playback_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 10px;
                border: 2px solid #FFFFFF;
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: {COLORS['background']};
                border-radius: 4px;
                margin: 2px 0;
            }}
            QSlider::handle:horizontal {{
                background: #FFD090;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
                border: 2px solid #FFFFFF;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent2']};
                border-radius: 4px;
            }}
            QLabel {{
                color: {COLORS['text']};
                font-weight: bold;
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
                color: {COLORS['text']};
                margin: 5px;
                padding: 0px;
            }}
            QProgressBar::chunk {{
                background-color: #B0E0B0;
                width: 10px;
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
            # 选择模型大小
            model_size = self.select_model_size()
            if not model_size:
                self.statusBar().showMessage('模型加载取消')
                return
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # 检查是否选择了CPU模式
            use_cpu = False
            if model_size.endswith("_cpu"):
                use_cpu = True
                model_size = model_size.replace("_cpu", "")
                logger.info(f"Loading SAM2 model: {model_size} in CPU mode")
                self.statusBar().showMessage(f'正在使用CPU模式加载模型 {model_size}...')
            else:
                logger.info(f"Loading SAM2 model: {model_size}")
            
            # 获取模型信息
            model_info = MODEL_INFO[model_size]
            
            # 检查本地模型文件
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
                    if not download_model(model_info, model_path, self):
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
            
            try:
                # 设置一些环境变量，避免可能的错误
                os.environ["TORCH_COMPILE_DISABLE_CUDA_GRAPHS"] = "1"
                os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
                
                # 设置设备
                device = "cpu" if use_cpu else "cuda"
                
                # 从本地文件加载
                if valid_local_file:
                    # 修复本地模型加载 - 需要同时提供配置文件和模型权重
                    logger.info(f"Loading from local file: {model_path}")
                    
                    # 根据模型文件名确定对应的配置文件
                    config_name = None
                    for size_key, info in MODEL_INFO.items():
                        if model_path.endswith(info['file_name']):
                            if "2.1" in info['file_name']:  # SAM 2.1 模型
                                config_name = f"configs/sam2.1/sam2.1_hiera_{size_key[0]}.yaml"
                                if size_key == "base":
                                    config_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                                break
                            else:  # SAM 2.0 模型
                                config_name = f"configs/sam2/sam2_hiera_{size_key[0]}.yaml"
                                if size_key == "base":
                                    config_name = "configs/sam2/sam2_hiera_b+.yaml"
                                break
                    
                    if not config_name:
                        # 如果无法匹配配置，根据文件名推断
                        if "tiny" in model_path:
                            config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
                        elif "small" in model_path:
                            config_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
                        elif "base_plus" in model_path or "base+" in model_path:
                            config_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                        elif "large" in model_path:
                            config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
                        else:
                            # 默认使用tiny配置
                            config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
                    
                    logger.info(f"Using config: {config_name} for model: {model_path}")
                    
                    # 导入构建函数
                    from sam2.build_sam import build_sam2_video_predictor
                    
                    # 使用build_sam2_video_predictor直接从本地文件加载
                    self.predictor = build_sam2_video_predictor(
                        config_file=config_name,
                        ckpt_path=model_path,
                        vos_optimized=False,
                        device=device
                    )
                else:
                    # 如果本地文件不存在，使用预训练模型ID
                    logger.info(f"Loading from Hugging Face model: {model_info['repo']}")
                    from sam2.build_sam import build_sam2_video_predictor_hf
                    self.predictor = build_sam2_video_predictor_hf(
                        model_id=model_info['repo'], 
                        vos_optimized=False,
                        device=device
                    )
                
                load_success = True
                
            except RuntimeError as e:
                # 处理CUDA内存不足错误
                if "CUDA out of memory" in str(e) or "device-side assert" in str(e):
                    # 尝试在CPU上重新加载tiny模型
                    retry = QMessageBox.question(
                        self, 
                        "GPU内存不足", 
                        f"您的GPU内存不足以加载{model_size}模型。\n\n是否尝试使用CPU模式加载tiny模型？\n(注意：CPU模式会非常慢)",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if retry == QMessageBox.Yes:
                        # 清理之前的尝试
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 使用CPU模式加载tiny模型
                        try:
                            tiny_model_info = MODEL_INFO["tiny"]
                            tiny_model_path = os.path.join(model_dir, tiny_model_info["file_name"])
                            
                            # 检查是否需要下载tiny模型
                            if not os.path.exists(tiny_model_path) or os.path.getsize(tiny_model_path) == 0:
                                self.statusBar().showMessage(f'正在下载tiny模型...')
                                if not download_model(tiny_model_info, tiny_model_path, self):
                                    self.progress_bar.setVisible(False)
                                    return
                            
                            self.statusBar().showMessage('正在使用CPU模式加载tiny模型...')
                            
                            # 使用正确的本地模型加载方法
                            from sam2.build_sam import build_sam2_video_predictor
                            self.predictor = build_sam2_video_predictor(
                                config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
                                ckpt_path=tiny_model_path,
                                vos_optimized=False,
                                device="cpu"
                            )
                            
                            load_success = True
                            model_size = "tiny"  # 更新模型大小
                            
                        except Exception as cpu_error:
                            error_messages.append(f"CPU模式加载失败: {str(cpu_error)}")
                        else:
                            error_messages.append("用户取消了CPU模式加载")
                    else:
                        error_messages.append(f"运行时错误: {str(e)}")
                else:
                    error_messages.append(f"运行时错误: {str(e)}")
            except Exception as e:
                error_messages.append(f"加载错误: {str(e)}")
            
            # 根据加载结果显示消息
            if load_success:
                self.statusBar().showMessage(f'模型 {model_size} 加载成功')
                self.progress_bar.setValue(100)
                QMessageBox.information(self, "成功", f"SAM2 模型 ({model_size}) 已成功加载！")
            else:
                error_details = "\n".join(error_messages)
                QMessageBox.critical(self, "错误", f"无法加载模型，请尝试重新下载。\n\n错误详情:\n{error_details}")
                self.statusBar().showMessage('模型加载失败')
            
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"加载模型过程中出错: {str(e)}")
            self.statusBar().showMessage('模型加载失败')
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
    
    def clear_prompts(self):
        """清除所有提示"""
        # 调用画布的清除方法
        if hasattr(self, 'video_canvas') and self.video_canvas is not None:
            self.video_canvas.clear_prompts()
            self.statusBar().showMessage('已清除所有提示')
        else:
            logger.warning("Cannot clear prompts, video_canvas not initialized")
    
    def toggle_prompt_type(self):
        """切换提示类型"""
        if self.point_radio.isChecked():
            self.video_canvas.set_point_mode(True)
            self.semantic_options_group.setVisible(False)
        elif self.box_radio.isChecked():
            self.video_canvas.set_point_mode(False)
            self.semantic_options_group.setVisible(False)
        elif self.semantic_radio.isChecked():
            # 激活语义分割模式
            self.video_canvas.set_semantic_mode(True)
            self.semantic_options_group.setVisible(True)
        
        logger.info(f"Prompt type toggled: point={self.point_radio.isChecked()}, box={self.box_radio.isChecked()}, semantic={self.semantic_radio.isChecked()}")
    
    def toggle_point_type(self):
        """切换点类型"""
        self.video_canvas.set_foreground_point(self.foreground_radio.isChecked())
    
    def update_threshold_value(self):
        """更新阈值显示"""
        value = self.threshold_slider.value()
        self.threshold_value_label.setText(f"{value}%")
        self.semantic_threshold = value / 100.0
        logger.info(f"Semantic threshold updated to {self.semantic_threshold}")
    
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
    
    def select_model_size(self):
        """选择模型大小"""
        dialog = QDialog(self)
        dialog.setWindowTitle("选择模型大小")
        dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # 添加说明标签
        label = QLabel("请根据您的硬件配置选择合适的模型大小:")
        label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(label)
        
        # 添加模型选项
        model_buttons = {}
        for size, info in MODEL_INFO.items():
            text = f"{info['name']} - {info['description']}"
            radio = QRadioButton(text)
            if size == "base":  # 默认选中base模型
                radio.setChecked(True)
            layout.addWidget(radio)
            model_buttons[size] = radio
        
        # 添加CPU选项
        cpu_checkbox = QCheckBox("使用CPU模式 (适用于无GPU或GPU内存不足的情况)")
        layout.addWidget(cpu_checkbox)
        
        layout.addSpacing(20)
        
        # 添加警告标签
        warning = QLabel("注意: 较大的模型需要更多内存和计算资源。如果您的电脑配置不足，可能会导致程序崩溃。")
        warning.setStyleSheet("color: red; font-style: italic;")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        
        # 添加按钮盒子
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # 显示对话框
        result = dialog.exec_()
        if result == QDialog.Accepted:
            for size, button in model_buttons.items():
                if button.isChecked():
                    selected = size
                    break
            else:
                selected = "base"  # 默认base
            
            # 添加CPU后缀
            if cpu_checkbox.isChecked():
                selected += "_cpu"
                
            return selected
        else:
            return None

    def add_segmentation(self):
        """添加分割"""
        if self.predictor is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if self.video_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先加载视频")
            return
        
        try:
            self.statusBar().showMessage('正在进行分割...')
            
            # 获取当前帧
            current_frame = self.video_canvas.current_frame
            frame_pil = cv2_to_pil(current_frame)
            
            # 获取提示数据
            point_coords, point_labels, box = self.video_canvas.get_prompt_data()
            
            # 判断分割模式
            if self.semantic_radio.isChecked():
                # 语义分割模式 - 自动检测所有物体
                self.statusBar().showMessage('正在进行语义分割...')
                masks = self.predictor.predict_semantic_masks(
                    frame_pil, 
                    threshold=self.semantic_threshold
                )
            else:
                # 提示分割模式 - 需要点或框
                if point_coords is None and box is None:
                    QMessageBox.warning(self, "警告", "请先添加点或框提示")
                    self.statusBar().showMessage('请添加提示')
                    return
                
                # 进行分割
                masks = self.predictor.predict_masks(
                    frame_pil,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    boxes=box.reshape(1, 4) if box is not None else None
                )
            
            # 添加分割结果
            self.video_canvas.add_segmentation_result(self.video_canvas.frame_index, masks)
            self.statusBar().showMessage('分割完成')
            
        except Exception as e:
            logger.error(f"Error in add_segmentation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"分割过程中出错: {str(e)}")
            self.statusBar().showMessage('分割失败')
    
    def track_objects(self):
        """追踪对象"""
        if self.predictor is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if self.video_canvas.current_frame is None:
            QMessageBox.warning(self, "警告", "请先加载视频")
            return
        
        try:
            # 检查当前帧是否有分割结果
            if self.video_canvas.frame_index not in self.video_canvas.segmentation_results:
                reply = QMessageBox.question(
                    self, 
                    "提示", 
                    "当前帧没有分割结果，是否先进行分割？",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.add_segmentation()
                else:
                    return
            
            # 如果正在追踪，则停止追踪
            if self.is_tracking:
                self.statusBar().showMessage('停止追踪')
                self.is_tracking = False
                return
            
            # 显示进度对话框
            progress = QProgressDialog("正在追踪对象...", "取消", 0, self.video_canvas.total_frames - self.video_canvas.frame_index, self)
            progress.setWindowTitle("追踪进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            
            self.statusBar().showMessage('开始追踪...')
            
            # 获取当前帧的掩码
            init_masks = self.video_canvas.segmentation_results[self.video_canvas.frame_index]
            
            # 获取当前帧的PIL图像
            init_frame = cv2_to_pil(self.video_canvas.current_frame)
            
            # 开始追踪
            for frame_idx in range(self.video_canvas.frame_index + 1, self.video_canvas.total_frames):
                # 检查是否取消
                if progress.wasCanceled():
                    break
                
                # 更新进度
                progress.setValue(frame_idx - self.video_canvas.frame_index)
                
                # 获取下一帧
                self.video_canvas.set_frame(frame_idx)
                next_frame = cv2_to_pil(self.video_canvas.current_frame)
                
                # 追踪对象
                tracked_masks = self.predictor.track_masks(next_frame, init_masks, init_frame)
                
                # 添加分割结果
                self.video_canvas.add_segmentation_result(frame_idx, tracked_masks)
                
                # 更新进度
                QApplication.processEvents()
            
            # 完成追踪
            progress.setValue(self.video_canvas.total_frames - self.video_canvas.frame_index)
            self.statusBar().showMessage('追踪完成')
            
        except Exception as e:
            logger.error(f"Error in track_objects: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"追踪过程中出错: {str(e)}")
            self.statusBar().showMessage('追踪失败')


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