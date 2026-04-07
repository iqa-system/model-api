# 给图片加上不同种类的噪声
from PIL import Image, ImageFilter

def add_gaussian_blur(img: Image.Image, x: int = None, y: int = None, width: int = None, height: int = None, radius: float = 5.0) -> Image.Image:
    """
    在指定矩形区域加上高斯模糊. 如果未指定坐标和宽高，则直接作用于全图.
    
    :param img: 原始 PIL Image 对象
    :param x: 矩形区域起点横坐标 (为None时默认从0开始)
    :param y: 矩形区域起点纵坐标 (为None时默认从0开始)
    :param width: 矩形区域的宽 (为None时默认直到图片边缘)
    :param height: 矩形区域的高 (为None时默认直到图片边缘)
    :param radius: 高斯模糊的程度(半径参数，值越大越模糊)
    :return: 处理后的新图片对象
    """
    result = img.copy()
    
    x = 0 if x is None else x
    y = 0 if y is None else y
    width = img.width if width is None else width
    height = img.height if height is None else height
    
    # 限制边界，防止越界
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.width, x + width), min(img.height, y + height)
    
    if x1 >= x2 or y1 >= y2:
        return result  # 区域无效直接返回
        
    box = (x1, y1, x2, y2)
    region = result.crop(box)
    
    # 应用高斯模糊
    region = region.filter(ImageFilter.GaussianBlur(radius=radius))
    
    result.paste(region, box)
    return result


def add_mosaic(img: Image.Image, x: int = None, y: int = None, width: int = None, height: int = None, pixel_size: int = 10) -> Image.Image:
    """
    在指定矩形区域加上马赛克模糊. 如果未指定坐标和宽高，则直接作用于全图.
    
    :param img: 原始 PIL Image 对象
    :param x: 矩形区域起点横坐标 (为None时默认从0开始)
    :param y: 矩形区域起点纵坐标 (为None时默认从0开始)
    :param width: 矩形区域的宽 (为None时默认直到图片边缘)
    :param height: 矩形区域的高 (为None时默认直到图片边缘)
    :param pixel_size: 马赛克方块的边长大小(程度参数，值越大马赛克越大)
    :return: 处理后的新图片对象
    """
    result = img.copy()
    
    x = 0 if x is None else x
    y = 0 if y is None else y
    width = img.width if width is None else width
    height = img.height if height is None else height
    
    # 限制边界，防止越界
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.width, x + width), min(img.height, y + height)
    
    if x1 >= x2 or y1 >= y2:
        return result  # 区域无效直接返回
        
    box = (x1, y1, x2, y2)
    region = result.crop(box)
    real_width = x2 - x1
    real_height = y2 - y1
    
    # 兼容处理 PIL 版本常量
    resample_filter = getattr(Image, 'Resampling', Image).NEAREST
    
    # 实现马赛克: 先按比例缩小, 再用最近邻插值放大回原尺寸
    small_size = (max(1, real_width // pixel_size), max(1, real_height // pixel_size))
    region = region.resize(small_size, resample=resample_filter)
    region = region.resize((real_width, real_height), resample=resample_filter)
    
    result.paste(region, box)
    return result


if __name__ == '__main__':
    # 测试代码样例
    import os
    
    test_img_path = 'test_imgs/human.jpg'  # 替换为你的测试图片路径
    out_dir = 'test_imgs'
    
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path).convert('RGB')
        
        # 测试 1: 加高斯模糊 (如起点在100,100，区域大小200x200，模糊半径10)
        blur_img = add_gaussian_blur(img, x=100, y=10, width=80, height=90, radius=5.0)
        blur_img.save(os.path.join(out_dir, 'human_blur_partial.jpg'))
        
        # 测试 2: 加马赛克 (如起点在300,100，区域大小200x200，马赛克块大小15)
        mosaic_img = add_mosaic(img, x=100, y=10, width=80, height=90, pixel_size=15)
        mosaic_img.save(os.path.join(out_dir, 'human_mosaic_partial.jpg'))
        
        # 测试 3: 全图加模糊/马赛克
        # full_blur_img = add_gaussian_blur(img, radius=3.0)
        # full_blur_img.save(os.path.join(out_dir, 'human_blur.jpg'))
        
        # full_mosaic_img = add_mosaic(img, pixel_size=3)
        # full_mosaic_img.save(os.path.join(out_dir, 'human_mosaic.jpg'))
        
        print("测试图片生成完毕!")
