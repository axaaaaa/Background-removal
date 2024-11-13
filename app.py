from transformers import pipeline # 导入transformers库，用于加载模型
import gradio as gr # 导入gradio库，界面
def remove_background(image):
    """
    移除图片背景的主要函数
    参数:
        image: 输入的图片文件
    返回:
        output_image: 去除背景后的图片
    """
    # 初始化图像分割pipeline
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    
    # 处理图像并返回去除背景后的图像
    output_image = pipe(image)
    
    return output_image # 返回去除背景后的图像

# 创建Gradio界面
demo = gr.Interface(
    fn=remove_background, # 函数
    inputs=gr.Image(type="filepath"), # 输入
    outputs=gr.Image(type="pil"), # 输出
    title="背景移除工具", # 标题
    description="上传一张图片，自动移除背景", # 描述
    examples=["123.jpg"]  # 可以添加示例图片
)
# 启动Gradio应用
if __name__ == "__main__":
    demo.launch(share=True) # 启动Gradio应用
