import os
import sys
import time
import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, 
                             QGroupBox, QFileDialog, QMessageBox, QProgressBar, QFrame,
                             QGridLayout, QStackedLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor, QPainter

# 配置类 - 增加色彩相关参数
class Config:
    pretrained_model_name_or_path = "model/LCM-runwayml-stable-diffusion-v1-5"  # 替换为你的本地模型路径
    resolution = 512
    rank = 2
    lora_alpha = 16
    device = "cpu"  # 可根据需要改为"cuda"
    num_final_inference_steps = 100
    guidance_scale = 5.0  # 降低引导尺度，使色彩更自然
    contrast_factor = 1.0  # 对比度调整因子
    saturation_factor = 1.0  # 饱和度调整因子
    brightness_factor = 1.0  # 亮度调整因子

# 加载LoRA权重的函数
def load_lora_weights(unet, load_path):
    lora_state_dict = torch.load(load_path, map_location=torch.device(Config.device))
    unet.load_state_dict(lora_state_dict, strict=False)
    return unet

# 修复tokenizer加载问题的函数
def load_tokenizer_with_fix(model_path):
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer")
        )
        return tokenizer
    except Exception as e:
        print(f"加载tokenizer时出错: {e}")
        print("尝试修复tokenizer配置...")
        
        from transformers import CLIPTokenizerFast
        
        vocab_file = os.path.join(model_path, "tokenizer", "vocab.json")
        merges_file = os.path.join(model_path, "tokenizer", "merges.txt")
        
        if os.path.exists(vocab_file) and os.path.exists(merges_file):
            tokenizer = CLIPTokenizerFast(
                vocab_file=vocab_file,
                merges_file=merges_file,
                max_length=77,
                pad_token="!",
                additional_special_tokens=["<startoftext|>", "<endoftext|>"]
            )
            return tokenizer
        else:
            raise Exception(f"找不到tokenizer文件: {vocab_file} 或 {merges_file}")

# 图像色彩校正函数
def adjust_image_colors(image):
    """调整图像的色彩、对比度和饱和度，使其更自然"""
    # 调整对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(Config.contrast_factor)
    
    # 调整饱和度
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(Config.saturation_factor)
    
    # 调整亮度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(Config.brightness_factor)
    
    return image

# 模型加载类
class ModelLoader:
    def __init__(self, config, lora_model_path):
        self.config = config
        self.lora_model_path = lora_model_path
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipe = None
        
    def load_models(self):
        self.tokenizer = load_tokenizer_with_fix(self.config.pretrained_model_name_or_path)
        
        text_encoder_path = os.path.join(self.config.pretrained_model_name_or_path, "text_encoder")
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        
        vae_path = os.path.join(self.config.pretrained_model_name_or_path, "vae")
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        
        unet_path = os.path.join(self.config.pretrained_model_name_or_path, "unet")
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        self.unet = load_lora_weights(self.unet, self.lora_model_path)
        
        self.text_encoder.to(self.config.device)
        self.vae.to(self.config.device)
        self.unet.to(self.config.device)
        
        scheduler_path = os.path.join(self.config.pretrained_model_name_or_path, "scheduler")
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        
        self.pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        return self.pipe

# 生成线程类 - 增加色彩校正步骤
class GenerateThread(QThread):
    finished = pyqtSignal(Image.Image)
    error = pyqtSignal(str)
    progress_updated = pyqtSignal(int, float)  # 进度百分比, 剩余时间(秒)
    
    def __init__(self, pipe, animal_name, num_inference_steps, guidance_scale, 
                 contrast_factor, saturation_factor, brightness_factor):
        super().__init__()
        self.pipe = pipe
        self.animal_name = animal_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.brightness_factor = brightness_factor
        self.start_time = 0
        self.step_times = []
        
    def run(self):
        try:
            # 优化提示词，增加环境和光照描述以改善色彩
            prompt = (f"a high quality photo of a {self.animal_name}, natural lighting, "
                     f"realistic colors, in natural habitat, detailed texture")
            
            # 在终端显示开始信息
            print(f"\n开始生成 '{self.animal_name}' 的图像...")
            print(f"总步数: {self.num_inference_steps}")
            print("进度: [", end="", flush=True)
            
            with torch.no_grad():
                text_inputs = self.pipe.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                
                text_embeddings = self.pipe.text_encoder(text_input_ids.to(self.pipe.device))[0]
                
                max_length = text_input_ids.shape[-1]
                uncond_input = self.pipe.tokenizer(
                    [""],
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.pipe.device))[0]
                
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                
                latents = torch.randn(
                    (1, self.pipe.unet.config.in_channels, 
                     Config.resolution // 8, Config.resolution // 8),
                    generator=torch.Generator(device=Config.device),
                    device=Config.device,
                )
                
                self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=Config.device)
                
                self.start_time = time.time()
                self.step_times = []
                
                for i, t in enumerate(self.pipe.scheduler.timesteps):
                    step_start_time = time.time()
                    
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                    
                    with torch.no_grad():
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                    
                    step_time = time.time() - step_start_time
                    self.step_times.append(step_time)
                    
                    progress = int((i + 1) / self.num_inference_steps * 100)
                    
                    steps_remaining = self.num_inference_steps - (i + 1)
                    if len(self.step_times) >= 5:
                        avg_step_time = sum(self.step_times[-5:]) / 5
                    else:
                        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
                    remaining_time = avg_step_time * steps_remaining
                    
                    # 更新终端进度条
                    if progress % 5 == 0:  # 每5%更新一次终端进度
                        bar_length = 20
                        filled_length = int(bar_length * progress // 100)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"\r进度: [{bar}] {progress}%", end="", flush=True)
                    
                    self.progress_updated.emit(progress, remaining_time)
            
                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = self.pipe.vae.decode(latents).sample
                
                # 改进的图像后处理
                image = (image / 2 + 0.5).clamp(0, 1)  # 标准化到[0,1]范围
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                # 转换为8位图像
                image = (image[0] * 255).round().astype("uint8")
                image = Image.fromarray(image)
                
                # 应用色彩校正
                image = adjust_image_colors(image)
                
                if image.size != (Config.resolution, Config.resolution):
                    image = image.resize((Config.resolution, Config.resolution), Image.LANCZOS)
            
            # 完成终端进度显示
            print(f"\r进度: [████████████████████] 100% - 完成!")
            total_time = time.time() - self.start_time
            print(f"总耗时: {total_time:.2f}秒")
            
            self.finished.emit(image)
        except Exception as e:
            # 出错时清除进度条
            print("\n错误发生!")
            self.error.emit(str(e))

# 主窗口类 - 添加色彩调整控件和水印
class AnimalGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipe = None
        self.current_image = None
        self.initUI()
        
    def initUI(self):
        # 设置中文字体支持
        font = QFont()
        font.setFamily("SimHei")  # 使用黑体字体，确保中文正常显示
        font.setPointSize(10)
        self.setFont(font)
        
        self.setWindowTitle('动物图像生成器')
        self.setGeometry(100, 100, 1100, 800)  # 增加窗口高度以容纳新控件
        
        # 设置窗口图标
        self.set_app_icon()
        
        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        
        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 3)
        
        # 创建右侧图像显示区域
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 5)
        
        # 状态栏样式
        self.statusBar().setStyleSheet("background-color: #f0f0f0; color: #333; padding: 5px;")
        
    def set_app_icon(self):
        """设置应用程序图标"""
        # 尝试多种路径查找图标
        possible_paths = [
            "icons/app_icon.png",
            "./icons/app_icon.png",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.setWindowIcon(QIcon(path))
                print(f"成功加载应用图标: {path}")
                return
        
        # 如果找不到图标文件，不报错，只在控制台提示
        print("未找到app_icon.png，使用默认图标")
    
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 添加标题
        title_label = QLabel("动物图像生成器")
        title_label.setFont(QFont("SimHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0;")
        layout.addWidget(line)
        
        # 模型设置区域 - 与之前相同
        model_group = QGroupBox("模型设置")
        self.style_group_box(model_group)
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(15, 15, 15, 15)
        model_layout.setSpacing(12)
        
        # LoRA模型路径
        lora_layout = QHBoxLayout()
        lora_layout.setSpacing(8)
        lora_label = QLabel("LoRA模型:")
        lora_label.setStyleSheet("font-weight: 500;")
        self.model_path_edit = QLineEdit()
        self.style_line_edit(self.model_path_edit)
        self.browse_btn = QPushButton("浏览")
        self.style_button(self.browse_btn, "browse")
        self.browse_btn.clicked.connect(self.browse_model)
        lora_layout.addWidget(lora_label)
        lora_layout.addWidget(self.model_path_edit)
        lora_layout.addWidget(self.browse_btn)
        model_layout.addLayout(lora_layout)
        
        # 基础模型路径
        base_model_layout = QHBoxLayout()
        base_model_layout.setSpacing(8)
        base_label = QLabel("基础模型:")
        base_label.setStyleSheet("font-weight: 500;")
        self.base_model_edit = QLineEdit(Config.pretrained_model_name_or_path)
        self.style_line_edit(self.base_model_edit)
        self.browse_base_btn = QPushButton("浏览")
        self.style_button(self.browse_base_btn, "browse")
        self.browse_base_btn.clicked.connect(self.browse_base_model)
        base_model_layout.addWidget(base_label)
        base_model_layout.addWidget(self.base_model_edit)
        base_model_layout.addWidget(self.browse_base_btn)
        model_layout.addLayout(base_model_layout)
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.setSpacing(8)
        device_label = QLabel("运行设备:")
        device_label.setStyleSheet("font-weight: 500;")
        self.device_edit = QLineEdit(Config.device)
        self.device_edit.setPlaceholderText("cpu 或 cuda")
        self.style_line_edit(self.device_edit)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_edit)
        model_layout.addLayout(device_layout)
        
        # 分辨率设置
        resolution_layout = QHBoxLayout()
        resolution_layout.setSpacing(8)
        res_label = QLabel("分辨率:")
        res_label.setStyleSheet("font-weight: 500;")
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(256, 1024)
        self.resolution_spin.setSingleStep(64)
        self.resolution_spin.setValue(Config.resolution)
        self.style_spin_box(self.resolution_spin)
        resolution_layout.addWidget(res_label)
        resolution_layout.addWidget(self.resolution_spin)
        model_layout.addLayout(resolution_layout)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.style_button(self.load_model_btn, "primary")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        layout.addWidget(model_group)
        
        # 生成参数区域
        generate_group = QGroupBox("生成参数")
        self.style_group_box(generate_group)
        generate_layout = QVBoxLayout(generate_group)
        generate_layout.setContentsMargins(15, 15, 15, 15)
        generate_layout.setSpacing(12)
        
        # 动物名称输入
        animal_layout = QHBoxLayout()
        animal_layout.setSpacing(8)
        animal_label = QLabel("动物名称:")
        animal_label.setStyleSheet("font-weight: 500;")
        self.animal_edit = QLineEdit()
        self.animal_edit.setPlaceholderText("请输入英文名称: lion, tiger, elephant")
        self.style_line_edit(self.animal_edit)
        animal_layout.addWidget(animal_label)
        animal_layout.addWidget(self.animal_edit)
        generate_layout.addLayout(animal_layout)
        
        # 推理步数
        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(8)
        steps_label = QLabel("推理步数:")
        steps_label.setStyleSheet("font-weight: 500;")
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 400)
        self.steps_spin.setValue(Config.num_final_inference_steps)
        self.style_spin_box(self.steps_spin)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_spin)
        generate_layout.addLayout(steps_layout)
        
        # 引导尺度 - 降低默认值
        guidance_layout = QHBoxLayout()
        guidance_layout.setSpacing(8)
        guidance_label = QLabel("引导尺度:")
        guidance_label.setStyleSheet("font-weight: 500;")
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(5.0)  # 降低默认引导尺度
        self.guidance_spin.setSingleStep(0.5)
        self.style_spin_box(self.guidance_spin)
        guidance_layout.addWidget(guidance_label)
        guidance_layout.addWidget(self.guidance_spin)
        generate_layout.addLayout(guidance_layout)
        
        generate_layout.addWidget(self.create_divider())
        
        # 新增：色彩调整参数
        color_label = QLabel("色彩调整:")
        color_label.setStyleSheet("font-weight: 500; margin-top: 5px;")
        generate_layout.addWidget(color_label)
        
        # 对比度调整
        contrast_layout = QHBoxLayout()
        contrast_layout.setSpacing(8)
        contrast_label = QLabel("对比度:")
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.1, 3.0)
        self.contrast_spin.setValue(Config.contrast_factor)
        self.contrast_spin.setSingleStep(0.1)
        self.style_spin_box(self.contrast_spin)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_spin)
        generate_layout.addLayout(contrast_layout)
        
        # 饱和度调整
        saturation_layout = QHBoxLayout()
        saturation_layout.setSpacing(8)
        saturation_label = QLabel("饱和度:")
        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setRange(0.1, 3.0)
        self.saturation_spin.setValue(Config.saturation_factor)
        self.saturation_spin.setSingleStep(0.1)
        self.style_spin_box(self.saturation_spin)
        saturation_layout.addWidget(saturation_label)
        saturation_layout.addWidget(self.saturation_spin)
        generate_layout.addLayout(saturation_layout)
        
        # 亮度调整
        brightness_layout = QHBoxLayout()
        brightness_layout.setSpacing(8)
        brightness_label = QLabel("亮度:")
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.1, 3.0)
        self.brightness_spin.setValue(Config.brightness_factor)
        self.brightness_spin.setSingleStep(0.1)
        self.style_spin_box(self.brightness_spin)
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_spin)
        generate_layout.addLayout(brightness_layout)
        
        # 生成按钮
        self.generate_btn = QPushButton("生成图像")
        self.style_button(self.generate_btn, "primary")
        self.generate_btn.clicked.connect(self.generate_image)
        self.generate_btn.setEnabled(False)
        generate_layout.addWidget(self.generate_btn)
        
        # 保存按钮
        self.save_btn = QPushButton("保存图像")
        self.style_button(self.save_btn, "secondary")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        generate_layout.addWidget(self.save_btn)
        
        layout.addWidget(generate_group)
        
        # 进度区域
        progress_group = QGroupBox("生成进度")
        self.style_group_box(progress_group)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(15, 15, 15, 15)
        progress_layout.setSpacing(10)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.style_progress_bar(self.progress_bar)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("等待生成...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #555; font-style: italic;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        # 添加拉伸项，使内容顶部对齐
        layout.addStretch()
        
        return panel
    
    def create_divider(self):
        """创建分隔线"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0; margin: 5px 0;")
        return line
    
    def create_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 图像显示区域标题
        title_layout = QHBoxLayout()
        title_label = QLabel("生成的图像")
        title_label.setFont(QFont("SimHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # 图像显示区域 - 使用带阴影的容器
        image_container = QWidget()
        image_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 15px;
        """)
        image_layout = QVBoxLayout(image_container)
        
        # 创建默认图片标签（底层，70%不透明度）
        self.default_image_label = QLabel()
        self.default_image_label.setAlignment(Qt.AlignCenter)
        self.default_image_label.setMinimumSize(512, 512)
        
        # 创建图像显示标签（中间层，100%不透明度）
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setText("")  # 初始为空，不显示任何文本
        self.image_label.setStyleSheet("background-color: transparent;")  # 确保背景透明
        
        # 添加水印标签（顶层）
        self.watermark_label = QLabel("制作者：热心市民小周")
        self.watermark_label.setStyleSheet("""
            color: rgba(100, 100, 100, 150);  /* 半透明灰色 */
            font-size: 12px;
            padding: 5px;
            background-color: rgba(255, 255, 255, 100);  /* 轻微透明背景 */
            border-radius: 2px;
        """)
        self.watermark_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)  # 右下角对齐
        
        # 使用网格布局来定位水印在右下角
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(self.default_image_label, 0, 0)  # 底层默认图
        grid_layout.addWidget(self.image_label, 0, 0)         # 中间生成的图
        grid_layout.addWidget(self.watermark_label, 0, 0)     # 顶层水印
        
        # 创建一个容器来放置网格布局
        grid_container = QWidget()
        grid_container.setLayout(grid_layout)
        grid_container.setMinimumSize(512, 512)
        
        # 尝试加载默认图片
        self.load_default_image()
        
        image_layout.addWidget(grid_container)
        layout.addWidget(image_container, 1)
        
        return panel
    
    def load_default_image(self):
        """加载默认图片"""
        try:
            # 尝试多种路径查找图片
            possible_paths = [
                "icons/default_animal.png",
                "./icons/default_animal.png",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "default_animal.png")
            ]
            
            default_pixmap = None
            found_path = None
            
            # 检查所有可能的路径
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    default_pixmap = QPixmap(path)
                    if not default_pixmap.isNull():
                        break
            
            if default_pixmap and not default_pixmap.isNull() and found_path:
                print(f"成功加载默认图片: {found_path}")
                
                # 设置不透明度为70%
                transparent_pixmap = QPixmap(default_pixmap.size())
                transparent_pixmap.fill(Qt.transparent)
                
                painter = QPainter(transparent_pixmap)
                painter.setOpacity(0.7)  # 70% 不透明度
                painter.drawPixmap(0, 0, default_pixmap)
                painter.end()
                
                # 设置图片并确保标签可见
                self.default_image_label.setPixmap(transparent_pixmap.scaled(
                    512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.default_image_label.setVisible(True)
                
            else:
                # 显示找不到图片的信息，但不报错
                print("未找到默认图片，显示提示文本")
                self.default_image_label.setText("请生成动物图像\n或放置 default_animal.png 到 icons 文件夹")
                self.default_image_label.setStyleSheet("""
                    border: 1px dashed #ccc;
                    border-radius: 4px;
                    color: #999;
                    font-style: italic;
                    padding: 20px;
                """)
                self.default_image_label.setVisible(True)
                
        except Exception as e:
            print(f"加载默认图片时出错: {e}")
            self.default_image_label.setText("默认图片加载失败")
            self.default_image_label.setStyleSheet("""
                border: 1px dashed #ccc;
                border-radius: 4px;
                color: #999;
                font-style: italic;
            """)
            self.default_image_label.setVisible(True)
    
    # UI样式美化方法
    def style_group_box(self, group_box):
        group_box.setStyleSheet("""
            QGroupBox {
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding: 5px;
                font-weight: bold;
                color: #34495e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def style_line_edit(self, line_edit):
        line_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px 8px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
                outline: none;
            }
        """)
        line_edit.setMinimumHeight(30)
    
    def style_spin_box(self, spin_box):
        spin_box.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px 8px;
                background-color: white;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                border-color: #3498db;
                outline: none;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #ccc;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-left: 1px solid #ccc;
                border-top: 1px solid #ccc;
            }
        """)
        spin_box.setMinimumHeight(30)
    
    def style_button(self, button, style_type="default"):
        if style_type == "primary":
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #2471a3;
                }
                QPushButton:disabled {
                    background-color: #bdc3c7;
                    color: #ecf0f1;
                }
            """)
        elif style_type == "secondary":
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
                QPushButton:pressed {
                    background-color: #219653;
                }
                QPushButton:disabled {
                    background-color: #bdc3c7;
                    color: #ecf0f1;
                }
            """)
        elif style_type == "browse":
            button.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 10px;
                    font-size: 9pt;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
                QPushButton:pressed {
                    background-color: #707b7c;
                }
            """)
        else:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    color: #333;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 8px 12px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)
        
        button.setMinimumHeight(30)
        try:
            if button.text() == "加载模型":
                button.setIcon(QIcon("icons/load_model.png"))
            elif button.text() == "生成图像":
                button.setIcon(QIcon("icons/generate.png"))
            elif button.text() == "保存图像":
                button.setIcon(QIcon("icons/save.png"))
            elif button.text() == "浏览":
                button.setIcon(QIcon("icons/browse.png"))
                
            button.setIconSize(QSize(16, 16))
        except:
            pass
    
    def style_progress_bar(self, progress_bar):
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                height: 8px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
    
    # 功能方法
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择LoRA模型文件", "", "Model Files (*.safetensors *.pth *.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_base_model(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择基础模型目录"
        )
        if dir_path:
            self.base_model_edit.setText(dir_path)
            
    def load_model(self):
        model_path = self.model_path_edit.text()
        base_model_path = self.base_model_edit.text()
        device = self.device_edit.text().strip().lower()
        
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的LoRA模型文件")
            return
            
        if not base_model_path or not os.path.exists(base_model_path):
            QMessageBox.warning(self, "错误", "请选择有效的基础模型目录")
            return
            
        try:
            Config.pretrained_model_name_or_path = base_model_path
            Config.device = device
            Config.resolution = self.resolution_spin.value()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  #  indeterminate progress
            self.progress_label.setText("正在加载模型，请稍候...")
            
            self.load_model_btn.setEnabled(False)
            self.statusBar().showMessage("正在加载模型，请稍候...")
            
            # 在终端显示加载信息
            print("开始加载模型...")
            
            model_loader = ModelLoader(Config, model_path)
            self.pipe = model_loader.load_models()
            
            self.generate_btn.setEnabled(True)
            self.statusBar().showMessage("模型加载成功!")
            self.progress_label.setText("模型加载成功!")
            
            # 在终端显示加载完成信息
            print("模型加载成功!")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型时出错: {str(e)}")
            self.progress_label.setText("模型加载失败")
            # 在终端显示错误信息
            print(f"模型加载失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.load_model_btn.setEnabled(True)
            
    def generate_image(self):
        if not self.pipe:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return
            
        animal_name = self.animal_edit.text().strip()
        if not animal_name:
            QMessageBox.warning(self, "错误", "请输入动物名称")
            return
            
        Config.resolution = self.resolution_spin.value()
        
        num_inference_steps = self.steps_spin.value()
        guidance_scale = self.guidance_spin.value()
        
        # 获取色彩调整参数
        contrast_factor = self.contrast_spin.value()
        saturation_factor = self.saturation_spin.value()
        brightness_factor = self.brightness_spin.value()
        
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备生成(第一次加载请耐心等待哦)...")
        self.statusBar().showMessage("正在生成图像，请稍候...")
        
        self.gen_thread = GenerateThread(
            self.pipe, animal_name, num_inference_steps, guidance_scale,
            contrast_factor, saturation_factor, brightness_factor
        )
        self.gen_thread.finished.connect(self.on_generation_finished)
        self.gen_thread.error.connect(self.on_generation_error)
        self.gen_thread.progress_updated.connect(self.on_progress_updated)
        self.gen_thread.start()
    
    def on_progress_updated(self, progress, remaining_time):
        self.progress_bar.setValue(progress)
        if remaining_time < 60:
            time_str = f"{remaining_time:.1f}秒"
        else:
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}分{seconds}秒"
        self.progress_label.setText(f"生成中: {progress}% 完成，剩余约 {time_str}")
        self.statusBar().showMessage(f"生成进度: {progress}%")
        
    def on_generation_finished(self, image):
        self.current_image = image
        pixmap = self.pil2pixmap(image)
        
        # 设置生成的图片到顶层标签，完全覆盖默认图片
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("生成完成!")
        self.statusBar().showMessage("图像生成成功!")
        
    def on_generation_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"生成图像时出错: {error_msg}")
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("生成失败")
        self.statusBar().showMessage("生成失败")
        
    def save_image(self):
        if not self.current_image:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)"
        )
        
        if file_path:
            try:
                self.current_image.save(file_path)
                self.statusBar().showMessage(f"图像已保存到: {file_path}")
                print(f"图像已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像时出错: {str(e)}")
                print(f"保存图像时出错: {str(e)}")
                
    def pil2pixmap(self, pil_image):
        if pil_image.mode in ("RGBA", "LA"):
            background = Image.new(pil_image.mode[:-1], pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        elif pil_image.mode == "P":
            pil_image = pil_image.convert("RGB")
            
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        width, height = pil_image.size
        data = pil_image.tobytes("raw", "RGB")
        qim = QImage(data, width, height, 3 * width, QImage.Format_RGB888)
        return QPixmap.fromImage(qim)

# 主函数
def main():
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")  # 使用Fusion风格，跨平台一致性更好
    
    # 设置全局调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(50, 50, 50))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(50, 50, 50))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(50, 50, 50))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    app.setApplicationName("动物图像生成器")
    app.setApplicationVersion("1.0")
    
    window = AnimalGeneratorApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
