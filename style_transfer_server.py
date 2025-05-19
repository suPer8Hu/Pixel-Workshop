# 添加详细的导入和错误处理
try:
    import numpy as np
    print(f"NumPy版本: {np.__version__}")
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    import torchvision
    print(f"TorchVision版本: {torchvision.__version__}")
    from PIL import Image
    print("所有依赖项已成功导入")
except ImportError as e:
    print(f"导入错误: {e}") 