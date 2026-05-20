from PIL import Image, ImageDraw

def draw_black_trapezoid_on_image(image_path, output_path, top_left, top_right, bottom_left, bottom_right):
    """
    在图片的指定位置绘制黑色梯形遮挡区域
    
    参数:
    image_path: 输入图片的路径
    output_path: 处理后图片的保存路径
    top_left: 梯形上底左上角坐标 (x1, y1)
    top_right: 梯形上底右上角坐标 (x2, y2)
    bottom_left: 梯形下底左下角坐标 (x3, y3)
    bottom_right: 梯形下底右下角坐标 (x4, y4)
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        
        # 设置纯黑色
        black_color = (0, 0, 0)
        
        # 定义梯形的四个顶点坐标（按顺序排列）
        trapezoid_points = [
            top_left,    # 上底左
            top_right,   # 上底右
            bottom_right,# 下底右
            bottom_left  # 下底左
        ]
        
        # 绘制并填充黑色梯形
        draw.polygon(trapezoid_points, fill=black_color)
        
        # 保存处理后的图片
        img.save(output_path)
        print(f"黑色梯形遮挡已绘制完成，图片已保存至: {output_path}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
    except Exception as e:
        print(f"未知错误：{e}")

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 自定义梯形的四个顶点坐标（你可以根据需要修改这些数值）
    trapezoid_top_left = (280, 360)    # 上底左上角
    trapezoid_top_right = (435, 360)   # 上底右上角
    trapezoid_bottom_left = (110, 480) # 下底左下角
    trapezoid_bottom_right = (640, 480)# 下底右下角

    
    # 调用函数绘制黑色梯形
    draw_black_trapezoid_on_image(
        image_path="front_pick_bottle.png",        # 你的输入图片路径
        output_path="output_trapezoid.png", # 输出图片路径
        top_left=trapezoid_top_left,
        top_right=trapezoid_top_right,
        bottom_left=trapezoid_bottom_left,
        bottom_right=trapezoid_bottom_right
    )