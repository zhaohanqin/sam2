import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QRect, QPoint

# 定义颜色常量
COLORS = {
    "background": "#FFFFFF",      # 背景色
    "foreground": "#F5F5F5",      # 前景色
    "accent1": "#E1F5FE",         # 强调色1
    "accent2": "#F3E5F5",         # 强调色2
    "text": "#333333",            # 文字颜色
    "border": "#E0E0E0",          # 边框颜色
    "hover": "#F8F9FA",           # 悬停颜色
    "active": "#E3F2FD"           # 激活颜色
}

# 设置全局字体样式
FONT_FAMILY = "微软雅黑"
FONT_SIZE = 12
BUTTON_FONT_SIZE = 14

class ImageCanvas(QLabel):
    """用于显示图像和处理交互的画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            background-color: {COLORS['background']};
            border: 2px solid {COLORS['border']};
            border-radius: 12px;
        """)
        
        # 图像相关
        self.image = None
        self.display_image = None
        self.scaled_image = None
        self.clicked_point = None
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
    
    def load_image(self, image_path):
        """加载图像"""
        try:
            # 加载图像
            self.image = QImage(image_path)
            if self.image.isNull():
                raise Exception("无法加载图像")
            
            # 更新显示
            self.update_display()
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            return False
    
    def update_display(self):
        """更新显示的图像"""
        if self.image is None:
            return
        
        # 根据控件大小缩放图像
        self.scaled_image = self.image.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 创建显示图像
        self.display_image = QImage(self.scaled_image)
        
        # 如果有点击点，绘制它
        if self.clicked_point:
            painter = QPainter(self.display_image)
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawEllipse(self.clicked_point, 5, 5)
            painter.end()
        
        # 设置图像
        self.setPixmap(QPixmap.fromImage(self.display_image))
    
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
        x_ratio = self.image.width() / self.scaled_image.width()
        y_ratio = self.image.height() / self.scaled_image.height()
        
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
        
        # 记录点击点
        self.clicked_point = QPoint(
            event.pos().x() - (self.width() - self.scaled_image.width()) // 2,
            event.pos().y() - (self.height() - self.scaled_image.height()) // 2
        )
        
        # 更新显示
        self.update_display()
        
        # 通知父窗口
        if self.parent:
            self.parent.update_coordinate_label(pos)
    
    def resizeEvent(self, event):
        """控件大小改变事件"""
        super().resizeEvent(event)
        self.update_display()

class CoordinatePicker(QMainWindow):
    """坐标选择器主窗口"""
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('图片坐标选择器')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QLabel {{
                color: {COLORS['text']};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE}px;
            }}
        """)
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 创建顶部工具栏
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['foreground']};
                border-radius: 12px;
                padding: 10px;
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(15, 15, 15, 15)
        toolbar_layout.setSpacing(15)
        
        # 加载图像按钮
        load_btn = QPushButton("加载图像")
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent1']};
                color: {COLORS['text']};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['active']};
            }}
        """)
        toolbar_layout.addWidget(load_btn)
        
        # 坐标显示标签
        self.coord_label = QLabel("坐标: 未选择")
        self.coord_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['accent2']};
                color: {COLORS['text']};
                border-radius: 8px;
                padding: 10px 20px;
                font-family: {FONT_FAMILY};
                font-size: {BUTTON_FONT_SIZE}px;
                font-weight: bold;
            }}
        """)
        toolbar_layout.addWidget(self.coord_label)
        
        main_layout.addWidget(toolbar_widget)
        
        # 创建图像画布
        self.image_canvas = ImageCanvas(self)
        main_layout.addWidget(self.image_canvas, 1)
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        self.statusBar().setStyleSheet(f"""
            background-color: {COLORS['foreground']};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE}px;
            padding: 5px;
        """)
    
    def load_image(self):
        """加载图像"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", options=options
        )
        
        if file_path:
            if self.image_canvas.load_image(file_path):
                # 获取图片尺寸
                width = self.image_canvas.image.width()
                height = self.image_canvas.image.height()
                # 更新状态栏和标签
                self.statusBar().showMessage(f'已加载图像: {os.path.basename(file_path)} ({width}x{height})')
                self.coord_label.setText(f"图片尺寸: {width}x{height} | 坐标: 未选择")
    
    def update_coordinate_label(self, pos):
        """更新坐标标签"""
        if pos:
            # 获取图片尺寸
            width = self.image_canvas.image.width()
            height = self.image_canvas.image.height()
            self.coord_label.setText(f"图片尺寸: {width}x{height} | 坐标: ({pos[0]}, {pos[1]})")
            self.statusBar().showMessage(f'已选择坐标: ({pos[0]}, {pos[1]})')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoordinatePicker()
    window.show()
    sys.exit(app.exec_()) 