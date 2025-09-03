import os
import glob
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import openpyxl
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from io import BytesIO
from peft import LoraConfig, get_peft_model

# 参数配置 - 关键优化点
class Config:
    # 数据参数 - 减少数据量
    data_root = "/kaggle/input/animals/Animal/Animal"  # 动物数据集根路径
    output_dir = "/kaggle/working/output"  # 所有输出文件的目录
    lora_model_dir = os.path.join(output_dir, "lora_models")  # 保存LoRA模型的目录
    history_file = os.path.join(output_dir, "training_history.xlsx")  # 训练历史记录文件
    sample_output_dir = os.path.join(output_dir, "validation_samples")  # 验证样本输出目录
    evaluation_file = os.path.join(output_dir, "evaluation_results.xlsx")  # 评估结果文件
    comparison_dir = os.path.join(output_dir, "comparison_samples")  # 对比样本目录
    
    # 模型参数 - 降低分辨率
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"  # 使用SD 1.5作为基础模型
    resolution = 256  # 降低分辨率以减少计算量 (原为512)
    center_crop = True  # 中心裁剪
    random_flip = True  # 随机水平翻转 (数据增强)

    # LoRA参数 - 简化LoRA
    rank = 2  # 降低LoRA的秩 (原为4)
    lora_alpha = 16  # 降低LoRA的alpha值 (原为32)

    # 训练参数 - 关键优化
    train_batch_size = 1  # 增加批大小以减少梯度累积步数 (原为1)
    gradient_accumulation_steps = 4  # 减少梯度累积步数 (原为4)
    num_train_epochs = 10  # 稍微增加训练轮数 (原为4)
    learning_rate = 1e-5  # 显著降低学习率 (原为2e-4)
    lr_scheduler_type = "cosine_with_warmup"  # 学习率调度器类型
    lr_warmup_steps = 200  # 增加预热步数 (原为50)
    max_grad_norm = 0.5  # 加强梯度裁剪 (原为1.0)
    use_ema = True  # 启用EMA以提高稳定性 (原为False)
    gradient_checkpointing = True  # 梯度检查点 (节省显存)
    mixed_precision = "fp16"  # 混合精度训练

    # 早停参数 - 使用CLIP分数作为指标
    early_stopping_patience = 5  # 增加早停耐心 (原为2)
    early_stopping_delta = 0.02  # CLIP分数的最小改善值
    validation_split = 0.1  # 验证集比例

    # 验证参数
    num_validation_samples = 5  # 随机选择多少种动物进行验证生成
    num_inference_steps = 20  # 减少推理步数以加快验证 (原为30)
    num_final_inference_steps = 100  # 最终评估时使用更多步数
    guidance_scale = 7.5  # 指导尺度 (CFG)
    
    # 新增: 每类最大样本数
    max_samples_per_class = 100  # 限制每类动物使用的最大样本数 (原为300-400)
    
    # 评估参数
    num_evaluation_samples = 10  # 评估时生成的样本数量
    clip_model_name = "openai/clip-vit-base-patch32"  # CLIP模型名称

# 确保输出目录存在
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.lora_model_dir, exist_ok=True)
os.makedirs(Config.sample_output_dir, exist_ok=True)
os.makedirs(Config.comparison_dir, exist_ok=True)

# 1. 数据处理与准备 - 添加样本限制
class AnimalDataset(Dataset):
    def __init__(self, data_root, tokenizer, size=384, center_crop=True, random_flip=True, max_samples_per_class=100):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size  # 使用新的分辨率
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_samples_per_class = max_samples_per_class
        
        # 获取所有图像路径和对应的类别（动物名称）
        self.image_paths = []
        self.class_names = []
        
        # 假设子文件夹以动物英文名称命名
        subfolders = [f.name for f in os.scandir(data_root) if f.is_dir()]
        for class_name in subfolders:
            class_dir = os.path.join(data_root, class_name)
            image_files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                         glob.glob(os.path.join(class_dir, "*.png")) + \
                         glob.glob(os.path.join(class_dir, "*.jpeg"))
            
            # 限制每类样本数量
            if len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
                
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.class_names.append(class_name)
        
        # 为每个类别创建提示词模板
        self.prompt_templates = [
            "a photo of a {}",
            "a high quality image of a {}",
            "a clear picture of a {}",
            "a realistic image of a {}",
            "a cute {}",
            "a wild {} in its natural habitat",
            "a close-up of a {}"
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        class_name = self.class_names[idx]
        
        # 加载和预处理图像
        image = Image.open(image_path).convert("RGB")
        
        # 调整大小和中心裁剪
        if self.center_crop:
            # 保持宽高比的调整大小和中心裁剪
            image = self._center_crop(image)
        else:
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 随机水平翻转 (数据增强)
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 将图像转换为模型输入的张量 (-1 to 1)
        image_tensor = (torch.tensor(np.array(image).astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)
        
        # 为图像生成随机的提示词
        prompt_template = random.choice(self.prompt_templates)
        prompt = prompt_template.format(class_name)
        
        # 对提示词进行标记化
        tokenized_input = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized_input.input_ids.squeeze(0)
        
        return {
            "pixel_values": image_tensor,
            "input_ids": input_ids,
            "prompt": prompt,
            "class_name": class_name
        }
    
    def _center_crop(self, image):
        width, height = image.size
        new_size = min(width, height)
        
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        return image

# 早停机制 (PyTorch实现) - 使用CLIP分数作为指标
class EarlyStopping:
    def __init__(self, patience=3, delta=0.05, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, clip_score):
        if self.best_score is None:
            self.best_score = clip_score
        elif clip_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = clip_score
            self.counter = 0

# 为UNet准备LoRA的函数 - 使用peft库
def prepare_unet_for_lora(unet, rank=2, alpha=16):
    # 配置LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    
    # 应用LoRA到UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    return unet

# 计算验证CLIP分数的函数
def compute_validation_clip_score(config, unet, text_encoder, vae, tokenizer, device):
    # 获取所有动物类别
    animal_classes = [f.name for f in os.scandir(config.data_root) if f.is_dir()]
    
    # 随机选择验证用的动物
    selected_animals = random.sample(animal_classes, min(config.num_validation_samples, len(animal_classes)))
    print(f"Selected animals for validation CLIP score: {selected_animals}")
    
    # 创建生成管道
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(device)
    
    # 加载CLIP模型和处理器
    clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    
    clip_scores = []
    
    for animal in selected_animals:
        prompt = f"a high quality photo of a {animal}"
        
        # 生成图像
        with torch.autocast(device.type):
            image = pipe(
                prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.resolution,
                width=config.resolution
            ).images[0]
        
        # 计算CLIP Score
        with torch.no_grad():
            # 处理图像和文本
            inputs = clip_processor(
                text=[prompt], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # 获取特征
            outputs = clip_model(**inputs)
            
            # 计算相似度 (CLIP Score)
            logits_per_image = outputs.logits_per_image  # 图像-文本相似度
            clip_score = logits_per_image.item()
        
        print(f"Animal: {animal}, CLIP Score: {clip_score:.4f}")
        clip_scores.append(clip_score)
    
    avg_clip_score = np.mean(clip_scores)
    print(f"Average Validation CLIP Score: {avg_clip_score:.4f}")
    
    return avg_clip_score

# 2. 训练函数 (包含早停和历史记录)
def train_lora_with_earlystopping(config):
    # 初始化模型组件
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet"
    )
    
    # 添加LoRA适配器到UNet
    unet = prepare_unet_for_lora(unet, config.rank, config.lora_alpha)
    
    # 设置噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # 启用梯度检查点以节省显存
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # 将模型移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    
    # 设置优化器 (只优化LoRA参数)
    lora_params = []
    for name, param in unet.named_parameters():
        if param.requires_grad:  # 只优化需要梯度的参数
            lora_params.append(param)
    
    # 使用更稳定的优化器配置
    optimizer = torch.optim.AdamW(
        lora_params, 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # 准备数据集和数据加载器
    full_dataset = AnimalDataset(
        config.data_root, tokenizer, size=config.resolution, 
        center_crop=config.center_crop, random_flip=config.random_flip,
        max_samples_per_class=config.max_samples_per_class
    )
    
    # 分割训练集和验证集
    val_size = int(len(full_dataset) * config.validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=2
    )
    
    # 计算总训练步数
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    # 学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # 初始化早停 (使用CLIP分数作为指标)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, 
        delta=config.early_stopping_delta,
        verbose=True
    )
    
    # 创建Excel工作簿用于记录历史
    history_wb = Workbook()
    history_ws = history_wb.active
    history_ws.title = "Training History"
    history_ws.append(["Epoch", "Step", "Train Loss", "Validation Loss", "CLIP Score", "Learning Rate", "Best CLIP Score", "Gradient Norm"])
    
    # 训练循环
    global_step = 0
    best_clip_score = 0.0
    
    # 训练循环部分
    for epoch in range(config.num_train_epochs):
        unet.train()
        total_loss = 0
        optimizer.zero_grad()
        current_grad_norm = 0.0  # 初始化梯度范数
        
        for step, batch in enumerate(train_dataloader):
            # 将批次数据移动到设备
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # 将图像编码到潜在空间
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # 缩放因子
            
            # 采样噪声
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # 向潜在表示添加噪声
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 获取文本嵌入
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            # 预测噪声残差
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # 计算损失
            loss = F.mse_loss(noise_pred, noise, reduction="mean") / config.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
                
                # 计算梯度范数用于监控
                current_grad_norm = 0
                for p in lora_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        current_grad_norm += param_norm.item() ** 2
                current_grad_norm = current_grad_norm ** 0.5
                
                # 更新参数
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # 打印训练信息
            if global_step % 50 == 0:  # 减少打印频率
                avg_loss = total_loss / (step + 1)
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Grad Norm: {current_grad_norm:.6f}")
        
        # 每个epoch结束后计算验证损失和CLIP分数
        val_loss = compute_validation_loss(unet, vae, text_encoder, val_dataloader, noise_scheduler, device)
        avg_train_loss = total_loss / len(train_dataloader)
        
        print(f"Epoch {epoch} completed. Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # 计算CLIP分数
        clip_score = compute_validation_clip_score(config, unet, text_encoder, vae, tokenizer, device)
        
        # 记录到历史
        current_lr = lr_scheduler.get_last_lr()[0]
        history_ws.append([epoch, global_step, avg_train_loss, val_loss, clip_score, current_lr, best_clip_score, current_grad_norm])
        
        # 早停检查 (基于CLIP分数)
        early_stopping(clip_score)
        
        # 保存最佳模型
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            # 保存LoRA权重
            save_path = os.path.join(config.lora_model_dir, f"lora_weights_epoch_{epoch}.safetensors")
            save_lora_weights(unet, save_path)
            print(f"Saved best model with CLIP score: {best_clip_score:.4f}")
        
        # 保存训练历史
        history_wb.save(config.history_file)
        
        # 检查早停
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("Training completed!")
    return unet, text_encoder, vae, tokenizer

# 计算验证损失的函数
def compute_validation_loss(unet, vae, text_encoder, val_dataloader, noise_scheduler, device):
    unet.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            # 将批次数据移动到设备
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # 将图像编码到潜在空间
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215  # 缩放因子
            
            # 采样噪声
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # 向潜在表示添加噪声
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 获取文本嵌入
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            # 预测噪声残差
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # 计算损失
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss

# 保存LoRA权重的函数
def save_lora_weights(unet, save_path):
    # 获取所有LoRA参数
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_state_dict[name] = param.cpu().detach()
    torch.save(lora_state_dict, save_path)

# 加载LoRA权重的函数
def load_lora_weights(unet, load_path):
    lora_state_dict = torch.load(load_path)
    unet.load_state_dict(lora_state_dict, strict=False)
    return unet

# 创建对比图像的函数
def create_comparison_image(images, titles, output_path):
    # 确保有偶数个图像
    if len(images) % 2 != 0:
        images = images[:-1]
        titles = titles[:-1]
    
    # 计算网格大小
    cols = 2
    rows = len(images) // cols
    
    # 计算每个图像的大小
    img_width, img_height = images[0].size
    
    # 创建空白画布
    result_width = img_width * cols
    result_height = img_height * rows + 50 * rows  # 为标题留出空间
    
    result = Image.new('RGB', (result_width, result_height), color='white')
    draw = ImageDraw.Draw(result)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("Arial", 30)
    except:
        font = ImageFont.load_default()
    
    # 放置图像和标题
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        # 计算位置
        x = col * img_width
        y = row * (img_height + 50)
        
        # 放置图像
        result.paste(img, (x, y + 50))
        
        # 添加标题
        text_width = draw.textlength(title, font=font)
        text_x = x + (img_width - text_width) / 2
        draw.text((text_x, y + 10), title, fill='black', font=font)
    
    # 保存结果
    result.save(output_path)
    print(f"Comparison image saved at: {output_path}")

# 3. 验证和生成样本
def generate_validation_samples(config, unet, text_encoder, vae, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取所有动物类别
    animal_classes = [f.name for f in os.scandir(config.data_root) if f.is_dir()]
    
    # 随机选择5种动物
    selected_animals = random.sample(animal_classes, config.num_validation_samples)
    print(f"Selected animals for validation: {selected_animals}")
    
    # 创建生成管道
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        safety_checker=None,  # 禁用安全检查器以加快生成速度
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(device)
    
    # 生成每种动物的图像
    all_images = []
    all_titles = []
    
    for animal in selected_animals:
        prompt = f"a high quality photo of a {animal}"
        
        # 生成图像 (使用更多推理步数)
        with torch.autocast(device.type):
            image = pipe(
                prompt,
                num_inference_steps=config.num_final_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.resolution,
                width=config.resolution
            ).images[0]
        
        # 保存图像
        save_path = os.path.join(config.sample_output_dir, f"{animal}.png")
        image.save(save_path)
        print(f"Generated image for {animal} saved at {save_path}")
        
        # 添加到列表用于创建对比图像
        all_images.append(image)
        all_titles.append(animal)
    
    # 创建对比图像
    comparison_path = os.path.join(config.comparison_dir, "animal_comparison.png")
    create_comparison_image(all_images, all_titles, comparison_path)
    
    return selected_animals

# 4. 评估函数 - 使用CLIP Score评估生成质量
def evaluate_with_clip_score(config, unet, text_encoder, vae, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载CLIP模型和处理器
    clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    
    # 获取所有动物类别
    animal_classes = [f.name for f in os.scandir(config.data_root) if f.is_dir()]
    
    # 随机选择评估用的动物
    selected_animals = random.sample(animal_classes, config.num_evaluation_samples)
    print(f"Selected animals for evaluation: {selected_animals}")
    
    # 创建生成管道
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    ).to(device)
    
    # 存储评估结果
    evaluation_results = []
    
    for animal in selected_animals:
        prompt = f"a high quality photo of a {animal}"
        
        # 生成图像 (使用更多推理步数)
        with torch.autocast(device.type):
            image = pipe(
                prompt,
                num_inference_steps=config.num_final_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.resolution,
                width=config.resolution
            ).images[0]
        
        # 保存图像
        save_path = os.path.join(config.sample_output_dir, f"eval_{animal}.png")
        image.save(save_path)
        
        # 计算CLIP Score
        with torch.no_grad():
            # 处理图像和文本
            inputs = clip_processor(
                text=[prompt], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # 获取特征
            outputs = clip_model(**inputs)
            
            # 计算相似度 (CLIP Score)
            logits_per_image = outputs.logits_per_image  # 图像-文本相似度
            clip_score = logits_per_image.item()
        
        print(f"Animal: {animal}, CLIP Score: {clip_score:.4f}")
        evaluation_results.append({
            "animal": animal,
            "prompt": prompt,
            "clip_score": clip_score,
            "image_path": save_path
        })
    
    # 计算平均CLIP Score
    avg_clip_score = np.mean([result["clip_score"] for result in evaluation_results])
    print(f"Average CLIP Score: {avg_clip_score:.4f}")
    
    # 保存评估结果到Excel
    evaluation_wb = Workbook()
    evaluation_ws = evaluation_wb.active
    evaluation_ws.title = "Evaluation Results"
    evaluation_ws.append(["Animal", "Prompt", "CLIP Score", "Image Path"])
    
    for result in evaluation_results:
        evaluation_ws.append([result["animal"], result["prompt"], result["clip_score"], result["image_path"]])
    
    evaluation_ws.append([])
    evaluation_ws.append(["Average CLIP Score", avg_clip_score])
    
    evaluation_wb.save(config.evaluation_file)
    print(f"Evaluation results saved to {config.evaluation_file}")
    
    return evaluation_results, avg_clip_score

# 主执行函数
def main():
    config = Config()
    
    # 训练模型
    print("Starting LoRA training...")
    unet, text_encoder, vae, tokenizer = train_lora_with_earlystopping(config)
    
    # 生成验证样本
    print("Generating validation samples...")
    generate_validation_samples(config, unet, text_encoder, vae, tokenizer)
    
    # 评估模型
    print("Evaluating model with CLIP Score...")
    evaluation_results, avg_clip_score = evaluate_with_clip_score(config, unet, text_encoder, vae, tokenizer)
    
    print(f"All done! Average CLIP Score: {avg_clip_score:.4f}")
    print("Check the output directory for results.")

if __name__ == "__main__":
    main()