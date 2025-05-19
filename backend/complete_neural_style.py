import os
import io
import sys
import base64
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageEnhance
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import gc
import random

app = Flask(__name__)
CORS(app)

# 首先确保所有依赖项已正确安装
try:
    import numpy as np
    import torch
    import torchvision
    DEPENDENCIES_OK = True
    print(f"依赖检查通过! NumPy: {np.__version__}, PyTorch: {torch.__version__}")
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f"缺少依赖项: {str(e)}")
    print("请执行: pip install numpy torch torchvision pillow flask flask-cors")

# 预训练的神经网络
def load_model():
    # 使用预训练的 VGG19 模型
    model = models.vgg19(pretrained=True).features.eval()
    return model

# 内容损失
def content_loss(content_weight, content_current, content_original):
    diff = content_current - content_original
    loss = content_weight * torch.sum(diff ** 2)
    return loss

# Gram矩阵计算
def gram_matrix(features, normalize=True):
    N, C, H, W = features.size()
    features_reshaped = features.view(N, C, -1)
    gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
    if normalize:
        gram /= (C * H * W)
    return gram

# 风格损失
def style_loss(feats, style_layers, style_targets, style_weights):
    style_loss = 0.0
    for i, layer_idx in enumerate(style_layers):
        current_feat = feats[layer_idx]
        current_gram = gram_matrix(current_feat)
        target_gram = style_targets[i]
        diff = current_gram - target_gram
        layer_loss = torch.sum(diff ** 2)
        style_loss += style_weights[i] * layer_loss
    return style_loss

# 总变差损失
def tv_loss(img, tv_weight):
    _, _, H, W = img.size()
    horizontal_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    horizontal_loss = torch.sum(horizontal_diff ** 2)
    vertical_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    vertical_loss = torch.sum(vertical_diff ** 2)
    loss = tv_weight * (horizontal_loss + vertical_loss)
    return loss

# 图像预处理
def preprocess(image, size=128):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 图像后处理
def postprocess(tensor):
    image = tensor.clone()
    
    # 反归一化
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # 裁剪值范围到[0,1]
    image = torch.clamp(image, 0, 1)
    
    # 转换为PIL图像
    image = image.permute(1, 2, 0).numpy() * 255
    return Image.fromarray(image.astype('uint8'))

# 提取特征
def extract_features(x, model):
    features = []
    for i, layer in enumerate(model.children()):
        x = layer(x)
        if i in {3, 8, 15, 22}:  # 选择特定的VGG层
            features.append(x)
    return features

# 图像从base64加载
def load_image_from_base64(base64_str):
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

# 完全重写神经风格迁移函数
def run_neural_style_transfer(content_img, style_img, image_size=128, iterations=100, 
                             content_weight=1.0, style_weight=1000000.0, tv_weight=1.0):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device.type}")
    
    if device.type == "cpu":
        print("警告: 未检测到CUDA设备，将使用CPU。这可能会很慢。")
    
    # 调整图像尺寸
    content_img = content_img.resize((image_size, image_size))
    style_img = style_img.resize((image_size, image_size))
    
    # 图像预处理
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)
    style_tensor = content_transform(style_img).unsqueeze(0).to(device)
    
    # 初始化目标图像（从内容图像复制）
    target = content_tensor.clone().requires_grad_(True)
    
    # 加载VGG19模型
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # 修改：使用具体的层名称而不是索引
    # 内容表示层 - 通常使用conv4_2
    content_layers = ['conv_4']  
    
    # 风格表示层 - 使用多个卷积层
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    # 创建层名称到网络索引的映射
    layer_mapping = {
        'conv_1': 0,
        'conv_2': 5,
        'conv_3': 10,
        'conv_4': 19,
        'conv_5': 28
    }
    
    # 修改特征提取函数以使用层名称映射
    def get_features(x, model):
        features = {}
        layer_count = 1
        conv_count = 1
        
        for i, layer in enumerate(model):
            x = layer(x)
            
            if isinstance(layer, nn.Conv2d):
                layer_name = f'conv_{conv_count}'
                if layer_name in content_layers or layer_name in style_layers:
                    features[layer_name] = x
                if i > 0 and isinstance(model[i-1], nn.MaxPool2d):
                    conv_count += 1
        
        return features
    
    # Gram矩阵计算
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    # 总变差损失
    def total_variation_loss(img):
        h_tv = torch.mean((img[:,:,1:,:] - img[:,:,:-1,:]).abs())
        w_tv = torch.mean((img[:,:,:,1:] - img[:,:,:,:-1]).abs())
        return h_tv + w_tv
    
    # 提取内容和风格特征
    content_features = get_features(content_tensor, vgg)
    style_features = get_features(style_tensor, vgg)
    
    # 计算风格的Gram矩阵
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}
    
    # 配置优化器
    optimizer = optim.Adam([target], lr=0.05)
    
    print_freq = 10
    
    n_iter = [0]
    
    while n_iter[0] < iterations:
        def closure():
            optimizer.zero_grad()
            
            # 获取当前目标图像的特征
            target_features = get_features(target, vgg)
            
            # 计算内容损失
            content_loss = 0
            for layer in content_layers:
                content_loss += torch.mean((target_features[layer] - content_features[layer]) ** 2)
            content_loss *= content_weight
            
            # 计算风格损失
            style_loss = 0
            for layer in style_layers:
                target_gram = gram_matrix(target_features[layer])
                style_loss += torch.mean((target_gram - style_grams[layer]) ** 2)
            style_loss *= style_weight
            
            # 总变差损失
            tv_loss = total_variation_loss(target) * tv_weight
            
            # 总损失
            loss = content_loss + style_loss + tv_loss
            
            # 反向传播 - 添加 retain_graph=True
            loss.backward(retain_graph=True)  # 关键修改
            
            n_iter[0] += 1
            
            # 打印损失
            if n_iter[0] % print_freq == 0 or n_iter[0] == 1:
                print(f"迭代 {n_iter[0]}: 内容损失: {content_loss.item():.4f}, 风格损失: {style_loss.item():.4f}")
            
            return loss
        
        optimizer.step(closure)
        
    # 后处理图像
    with torch.no_grad():
        target_image = target.clone().squeeze()
        # 反归一化
        target_image = target_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        target_image = target_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        # 裁剪到[0, 1]范围
        target_image = torch.clamp(target_image, 0, 1)
        # 转换为PIL图像
        target_image = target_image.cpu().permute(1, 2, 0).numpy() * 255.0
        output_img = Image.fromarray(target_image.astype('uint8'))
        
    # 保存输入图像和结果用于调试
    os.makedirs("debug_images", exist_ok=True)
    content_img.save("debug_images/content_input.png")
    style_img.save("debug_images/style_input.png")
    output_img.save("debug_images/result.png")
        
    return output_img

@app.route('/api/style-transfer', methods=['POST'])
def style_transfer_api():
    if not DEPENDENCIES_OK:
        return jsonify({'error': '服务器缺少必要的依赖项'}), 500
    
    try:
        print("===== 风格迁移开始 =====")
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求中没有数据'}), 400
            
        content_base64 = data.get('contentImage', '')
        style_base64 = data.get('styleImage', '')
        preset = data.get('preset', 'balanced')
        canvasSize = int(data.get('canvasSize', 16))
        
        # 限制处理大小
        max_input_size = 256  # 最大输入尺寸
        if canvasSize > max_input_size:
            print(f"警告: 画布尺寸({canvasSize})超过最大处理尺寸，将限制为{max_input_size}")
            canvasSize = max_input_size
        
        if not content_base64 or not style_base64:
            return jsonify({'error': '缺少必要的图像数据'}), 400
        
        # 解码内容图像
        try:
            print("解码内容图像...")
            if "base64," in content_base64:
                content_base64 = content_base64.split("base64,")[1]
            content_bytes = base64.b64decode(content_base64)
            content_img = Image.open(io.BytesIO(content_bytes))
            print(f"内容图像成功解码: {content_img.size} ({content_img.mode})")
            
            # 检查图像是否全黑
            is_black = True
            pixels = content_img.convert('RGB').getdata()
            for pixel in pixels:
                if pixel != (0, 0, 0) and pixel != (0, 0, 0, 0) and pixel != (0, 0, 0, 255):
                    is_black = False
                    print(f"找到非黑色像素: {pixel}")
                    break
            
            if is_black:
                print("警告: 内容图像似乎是全黑的！提供替代图像。")
                # 创建一个彩色测试图像
                content_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
                draw = ImageDraw.Draw(content_img)
                for i in range(20):
                    x1, y1 = random.randint(0, 200), random.randint(0, 200)
                    x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
                    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                    draw.rectangle((x1, y1, x2, y2), fill=color)
                
        except Exception as e:
            print(f"内容图像解码失败: {str(e)}")
            return jsonify({'error': f'内容图像解码失败: {str(e)}'}), 400
            
        # 解码风格图像
        try:
            print("解码风格图像...")
            if "base64," in style_base64:
                style_base64 = style_base64.split("base64,")[1]
            style_bytes = base64.b64decode(style_base64)
            style_img = Image.open(io.BytesIO(style_bytes))
            print(f"风格图像成功解码: {style_img.size} ({style_img.mode})")
        except Exception as e:
            print(f"风格图像解码失败: {str(e)}")
            return jsonify({'error': f'风格图像解码失败: {str(e)}'}), 400
        
        # 转换图像模式
        if content_img.mode != 'RGB':
            print(f"将内容图像从 {content_img.mode} 转换到 RGB")
            content_img = content_img.convert('RGB')
        if style_img.mode != 'RGB':
            print(f"将风格图像从 {style_img.mode} 转换到 RGB")
            style_img = style_img.convert('RGB')
        
        # 调试保存
        os.makedirs("debug_images", exist_ok=True)
        content_save_path = "debug_images/content_input.png"
        style_save_path = "debug_images/style_input.png"
        content_img.save(content_save_path)
        style_img.save(style_save_path)
        
        # 根据画布尺寸调整处理参数
        if canvasSize > 128:
            # 特大画布 - 使用最快的设置
            image_size = 256  # 使用更大的处理尺寸以保留细节
            content_weight = 0.05
            style_weight = 50000.0
            tv_weight = 1e-2
            num_steps = 120  # 减少迭代次数，避免处理时间过长
        elif canvasSize > 64:
            # 大画布 - 平衡设置
            image_size = 256
            content_weight = 0.07
            style_weight = 70000.0
            tv_weight = 1e-2
            num_steps = 180
        else:
            # 小/中画布 - 可以使用更精细的设置
            image_size = 256
            content_weight = 0.1
            style_weight = 80000.0
            tv_weight = 1e-2
            num_steps = 200
            
        print(f"使用优化参数: 图像尺寸={image_size}, 内容权重={content_weight}, 风格权重={style_weight}, TV权重={tv_weight}, 迭代={num_steps}")
        
        # 调整图像大小 - 使用NEAREST而不是LANCZOS，保留像素边缘
        content_img = content_img.resize((image_size, image_size), Image.NEAREST)
        style_img = style_img.resize((image_size, image_size), Image.LANCZOS)  # 风格图像可以用LANCZOS
        
        # 调用风格迁移函数
        result_img = run_neural_style_transfer(
            content_img, 
            style_img,
            image_size=image_size,
            iterations=num_steps
        )
        
        # 在结果处理中考虑原始画布尺寸
        # 确保输出的图像尺寸匹配原始画布尺寸，以避免失真
        result_img = result_img.resize((canvasSize, canvasSize), Image.NEAREST)
        
        # 像素艺术特定的后处理
        # 1. 锐化显著增强
        enhancer = ImageEnhance.Sharpness(result_img)
        result_img = enhancer.enhance(2.0)  # 增加锐化强度
        
        # 2. 添加像素化处理来增强pixel art效果
        pixelate_size = canvasSize = int(data.get('canvasSize', 16))

        # 对于大尺寸画布，我们需要限制输出图像尺寸，避免内存问题
        if pixelate_size > 128:
            # 对于较大的画布，先渲染到128大小
            small_size = 128
            result_img = result_img.resize((small_size, small_size), Image.NEAREST)
            # 然后根据实际画布大小放大
            result_img = result_img.resize((pixelate_size, pixelate_size), Image.NEAREST)
            # 最后确保输出不超过256×256
            output_size = min(256, pixelate_size)
            result_img = result_img.resize((output_size, output_size), Image.NEAREST)
        else:
            # 小尺寸画布使用原来的处理逻辑
            small_size = pixelate_size
            result_img = result_img.resize((small_size, small_size), Image.NEAREST)
            result_img = result_img.resize((256, 256), Image.NEAREST)
        
        # 保存结果
        result_save_path = "debug_images/style_result.png"
        result_img.save(result_save_path)
        
        # 确保两个图像文件存在
        if not os.path.exists("debug_images/result.png") or not os.path.exists("debug_images/style_result.png"):
            # 如果缺少文件，可以生成两个不同的图像版本
            print("警告：缺少图像文件，生成备用图像")
            
            # 如果缺少第一个图像，创建并保存它
            if not os.path.exists("debug_images/result.png"):
                with open("debug_images/result.png", "wb") as f:
                    enhancer = ImageEnhance.Contrast(result_img)
                    alt_img = enhancer.enhance(1.5)
                    buffered = io.BytesIO()
                    alt_img.save(buffered, format="PNG")
                    f.write(buffered.getvalue())
        
        # 读取两个图像文件
        with open("debug_images/result.png", "rb") as f:
            result_bytes = f.read()
        with open("debug_images/style_result.png", "rb") as f:
            style_bytes = f.read()
            
        # 编码为base64
        result_base64 = base64.b64encode(result_bytes).decode()
        style_base64 = base64.b64encode(style_bytes).decode()
        
        print("===== 风格迁移完成 =====")
    
        return jsonify({
            'resultImage': f'data:image/png;base64,{result_base64}',  
            'alternativeImage': f'data:image/png;base64,{style_base64}'  
        })
        
    except Exception as e:
        print(f"风格迁移过程中出错: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    dependency_status = "✅ 所有依赖项已安装" if DEPENDENCIES_OK else "❌ 缺少依赖项"
    return f"""
    <html>
    <body>
        <h1>神经风格迁移服务器</h1>
        <p>服务器状态: {dependency_status}</p>
        <h2>可用的API端点:</h2>
        <ul>
            <li><strong>/api/style-transfer</strong> - 默认尺寸 (64x64)</li>
            <li><strong>/api/style-transfer-128</strong> - 中等尺寸 (128x128)</li>
            <li><strong>/api/style-transfer-256</strong> - 大尺寸 (256x256)</li>
        </ul>
    </body>
    </html>
    """

# 128x128 尺寸的风格迁移端点
@app.route('/api/style-transfer-128', methods=['POST'])
def style_transfer_api_128():
    if not DEPENDENCIES_OK:
        return jsonify({
            'error': '服务器缺少必要的依赖项。请检查服务器日志。'
        }), 500
    
    try:
        data = request.json
        content_base64 = data.get('contentImage', '')
        style_base64 = data.get('styleImage', '')
        
        if not content_base64 or not style_base64:
            return jsonify({'error': '缺少图像数据'}), 400
        
        # 加载图像
        content_img = load_image_from_base64(content_base64)
        style_img = load_image_from_base64(style_base64)
        
        print(f"进行128x128尺寸的风格迁移")
        
        # 执行风格迁移 - 使用128尺寸
        output_img = run_neural_style_transfer(
            content_img, 
            style_img, 
            image_size=128,
            iterations=200  # 128尺寸使用中等迭代次数
        )
        
        # 返回结果
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'resultImage': f'data:image/png;base64,{img_str}'})
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 256x256 尺寸的风格迁移端点
@app.route('/api/style-transfer-256', methods=['POST'])
def style_transfer_api_256():
    if not DEPENDENCIES_OK:
        return jsonify({
            'error': '服务器缺少必要的依赖项。请检查服务器日志。'
        }), 500
    
    try:
        data = request.json
        content_base64 = data.get('contentImage', '')
        style_base64 = data.get('styleImage', '')
        
        if not content_base64 or not style_base64:
            return jsonify({'error': '缺少图像数据'}), 400
        
        # 加载图像
        content_img = load_image_from_base64(content_base64)
        style_img = load_image_from_base64(style_base64)
        
        print(f"进行256x256尺寸的风格迁移")
        
        # 清理内存以提高成功率
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 执行风格迁移 - 使用256尺寸
        output_img = run_neural_style_transfer(
            content_img, 
            style_img, 
            image_size=256,
            iterations=200
        )
        
        # 返回结果
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'resultImage': f'data:image/png;base64,{img_str}'})
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU内存不足错误: {str(e)}")
            print("尝试减小图像尺寸或降低批处理大小")
            # 如果出现GPU内存不足，尝试使用CPU
            if device.type == "cuda":
                print("切换到CPU进行处理")
                torch.cuda.empty_cache()
                # 此处可以递归调用自身，但使用CPU模式
                # 或者返回错误信息
                raise Exception("GPU内存不足，请尝试较小的图像尺寸")
        else:
            raise e
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """
    检查服务器状态的简单端点
    """
    return jsonify({
        'status': 'online',
        'cuda_available': torch.cuda.is_available(),
        'dependencies_ok': DEPENDENCIES_OK
    })

# 在styles目录创建默认风格图像
def create_default_style_images():
    os.makedirs('styles', exist_ok=True)
    
    # 创建马赛克风格
    mosaic = Image.new('RGB', (256, 256), (240, 240, 240))
    for x in range(0, 256, 16):
        for y in range(0, 256, 16):
            color = ((x+y) % 256, (x*y//256) % 256, (x+y*2) % 256)
            ImageDraw.Draw(mosaic).rectangle([x, y, x+16, y+16], fill=color)
    mosaic.save('styles/mosaic.jpg')
    
    # 创建素描风格
    sketch = Image.new('RGB', (256, 256), (240, 240, 240))
    for i in range(20):
        x1, y1 = random.randint(0, 255), random.randint(0, 255)
        x2, y2 = random.randint(0, 255), random.randint(0, 255)
        width = random.randint(1, 3)
        ImageDraw.Draw(sketch).line([(x1, y1), (x2, y2)], fill=(10, 10, 10), width=width)
    sketch.save('styles/sketch.jpg')
    
    # 创建水彩风格
    watercolor = Image.new('RGB', (256, 256), (240, 240, 240))
    for i in range(50):
        x, y = random.randint(0, 255), random.randint(0, 255)
        r = random.randint(5, 30)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        ImageDraw.Draw(watercolor).ellipse([x-r, y-r, x+r, y+r], fill=color)
    watercolor.save('styles/watercolor.jpg')
    
    # 创建油画风格
    oil = Image.new('RGB', (256, 256), (220, 220, 200))
    for i in range(30):
        x1, y1 = random.randint(0, 255), random.randint(0, 255)
        width = random.randint(10, 40)
        height = random.randint(10, 40)
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        ImageDraw.Draw(oil).rectangle([x1, y1, x1+width, y1+height], fill=color)
    oil.save('styles/oil.jpg')

# 启动服务器时创建默认风格图像
create_default_style_images()

def simple_neural_style_transfer(content_img, style_img, content_weight=1, style_weight=1000000, tv_weight=1, num_steps=100):
    """
    简化版风格迁移，但使用更强的风格权重和不同的层选择
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 强制设置统一尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    
    # 正确创建目标张量 - 确保它是叶节点张量
    # 先创建噪声，然后与内容进行混合
    target = content_tensor.clone().detach()  # 确保是叶节点
    noise = torch.randn_like(target).to(device) * 0.1
    with torch.no_grad():  # 不计算混合操作的梯度
        target.add_(noise)  # 添加噪声
    target.requires_grad_(True)  # 然后启用梯度
    
    # 加载VGG19模型
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # 更改使用的层 - 这些是在VGG19中实际有效的层
    # 用于内容的深层特征
    content_layers = ['conv4_2'] 
    # 用于风格的多个层
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    # 层名称到索引的映射
    layer_mapping = {
        'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
        'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
        'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14,
        'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
        'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22, 'conv4_3': 23,
        'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
        'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31, 'conv5_3': 32,
        'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36
    }
    
    # 获取层索引
    content_layer_indices = [layer_mapping[name] for name in content_layers]
    style_layer_indices = [layer_mapping[name] for name in style_layers]
    
    # 给不同风格层分配权重
    style_weights = [1000000, 800000, 100000, 10000, 1000]  # 更高的权重
    
    # 提取特征
    def get_features(x):
        features = {}
        i = 0
        for layer in vgg.children():
            x = layer(x)
            if i in content_layer_indices or i in style_layer_indices:
                features[i] = x
            i += 1
        return features
    
    # 计算风格的Gram矩阵
    def gram_matrix(features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    # 提取内容特征和风格特征
    print("提取内容和风格特征...")
    content_features = get_features(content_tensor)
    style_features = get_features(style_tensor)
    
    # 内容表示
    content_targets = {idx: content_features[idx].detach() for idx in content_layer_indices}
    
    # 计算风格的Gram矩阵
    style_grams = {idx: gram_matrix(style_features[idx]).detach() for idx in style_layer_indices}
    
    # 总变差损失
    def tv_loss(img):
        h_tv = torch.sum((img[:,:,1:,:] - img[:,:,:-1,:]).abs())
        w_tv = torch.sum((img[:,:,:,1:] - img[:,:,:,:-1]).abs())
        return h_tv + w_tv
    
    # 优化
    optimizer = optim.Adam([target], lr=0.05)
    
    print("开始风格迁移优化...")
    for step in range(num_steps):
        # 清除梯度
        optimizer.zero_grad()
        
        # 获取当前目标图像的特征
        target_features = get_features(target)
        
        # 内容损失
        content_loss = 0
        for idx in content_layer_indices:
            content_loss += torch.mean((target_features[idx] - content_targets[idx]) ** 2)
        content_loss *= content_weight
        
        # 风格损失
        style_loss = 0
        for i, idx in enumerate(style_layer_indices):
            target_gram = gram_matrix(target_features[idx])
            style_gram = style_grams[idx]
            layer_style_loss = style_weights[i] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss
        
        # 总变差损失
        t_loss = tv_weight * tv_loss(target)
        
        # 总损失
        loss = content_loss + style_loss + t_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 确保值在有效范围内
        with torch.no_grad():
            target.clamp_(0, 1)
        
        # 每10步打印一次
        if step % 10 == 0 or step == num_steps - 1:
            print(f"迭代 {step}/{num_steps}: 内容损失={content_loss.item():.2f}, 风格损失={style_loss.item():.2f}")
    
    # 处理结果图像
    with torch.no_grad():
        output = target.clone().squeeze(0).cpu()
        # 反归一化
        output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output = output + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # 裁剪到[0, 1]范围
        output = torch.clamp(output, 0, 1)
        # 转换为PIL图像
        output = output.permute(1, 2, 0).numpy() * 255.0
        result_img = Image.fromarray(output.astype('uint8'))
        
    return result_img

@app.route('/test-style', methods=['GET'])
def test_style_transfer():
    """测试端点，使用本地图像执行风格迁移"""
    try:
        print("测试风格迁移功能...")
        
        # 加载测试图像
        content_path = "test_images/content.jpg"
        style_path = "test_images/style.jpg"
        
        if not (os.path.exists(content_path) and os.path.exists(style_path)):
            os.makedirs("test_images", exist_ok=True)
            # 创建简单的测试图像
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(content_path)
            # 创建简单的彩色风格图像
            style_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
            draw = ImageDraw.Draw(style_img)
            for i in range(10):
                x1, y1 = random.randint(0, 200), random.randint(0, 200)
                x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle((x1, y1, x2, y2), fill=color)
            style_img.save(style_path)
        
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")
        
        # 执行风格迁移
        result_img = simple_neural_style_transfer(
            content_img,
            style_img,
            content_weight=0.025,
            style_weight=1000000.0,
            tv_weight=0.1,
            num_steps=20  # 减少迭代次数加快测试
        )
        
        # 保存结果
        test_result_path = "test_images/result.jpg"
        result_img.save(test_result_path)
        
        # 返回HTML页面显示结果
        html = f"""
        <html>
        <head><title>风格迁移测试</title></head>
        <body>
            <h1>风格迁移测试</h1>
            <div style="display:flex">
                <div>
                    <h2>内容图像</h2>
                    <img src="/test-image/content.jpg" style="width:256px">
                </div>
                <div>
                    <h2>风格图像</h2>
                    <img src="/test-image/style.jpg" style="width:256px">
                </div>
                <div>
                    <h2>结果图像</h2>
                    <img src="/test-image/result.jpg" style="width:256px">
                </div>
            </div>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"测试失败: {str(e)}", 500

@app.route('/test-image/<filename>', methods=['GET'])
def test_image(filename):
    filepath = os.path.join(os.getcwd(), "test_images", filename)
    if not os.path.exists(filepath):
        return "文件不存在", 404
    return send_file(filepath)

@app.route('/api/simple-test', methods=['POST'])
def simple_test_api():
    """简单测试端点，不做任何风格迁移，只返回接收到的图像"""
    try:
        data = request.json
        content_base64 = data.get('contentImage', '')
        
        # 解码内容图像
        if "base64," in content_base64:
            content_base64 = content_base64.split("base64,")[1]
        content_bytes = base64.b64decode(content_base64)
        
        # 原样返回
        return jsonify({'resultImage': f'data:image/png;base64,{content_base64}'})
    except Exception as e:
        print(f"测试API失败: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动神经风格迁移服务器...")
    print(f"依赖项状态: {'已安装' if DEPENDENCIES_OK else '缺失'}")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    - 总内存: {props.total_memory / 1e9:.2f} GB")
            print(f"    - CUDA能力: {props.major}.{props.minor}")
    else:
        print("警告: 未检测到CUDA设备，将使用CPU。这会导致处理速度显著变慢。")
    
    app.run(host='0.0.0.0', port=5001, debug=True) 