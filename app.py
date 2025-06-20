import os
import sys
import torch
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import DigitCNN, AttentionDigitCNN
from utils import preprocess_cv_image, get_prediction, draw_attention_heatmap

class DigitCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=280, height=280, bg='white', **kwargs)
        self.parent = parent
        self.bind("<Button-1>", self.start_draw)
        self.bind("<B1-Motion>", self.draw)
        self.old_x = None
        self.old_y = None
        
        # 创建一个PIL图像用于保存绘制内容
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def start_draw(self, event):
        self.old_x = event.x
        self.old_y = event.y
    
    def draw(self, event):
        if self.old_x and self.old_y:
            self.create_line(self.old_x, self.old_y, event.x, event.y, 
                             width=20, fill='black', capstyle=tk.ROUND, smooth=True)
            # 同时在PIL图像上绘制
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill=0, width=20)
        self.old_x = event.x
        self.old_y = event.y
    
    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None
    
    def get_image(self):
        """获取当前画布的图像，转换为OpenCV格式"""
        # 从PIL图像转换为numpy数组
        img_array = np.array(self.image)
        return img_array


class PredictionVisualizer(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # 创建matplotlib图表
        self.figure, self.ax = plt.subplots(figsize=(4, 4))
        self.figure.patch.set_facecolor('#F0F0F0')  # 设置与tkinter背景相近的颜色
        
        # 创建tkinter的matplotlib容器
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化显示
        self.ax.set_title("Prediction will be displayed here")
        self.ax.axis('off')
        self.canvas.draw()
    
    def show_prediction(self, digit, probability):
        """显示预测结果"""
        self.ax.clear()
        # 创建条形图
        bars = self.ax.bar(range(10), [0.1] * 10, color='skyblue')
        # 突出显示预测的数字
        bars[digit].set_color('red')
        bars[digit].set_height(probability)
        
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Digit")
        self.ax.set_ylabel("Probability")
        self.ax.set_title(f"Prediction: {digit}\nConfidence: {probability:.2f}")
        
        self.canvas.draw()
    
    def show_image(self, img):
        """显示图像"""
        self.ax.clear()
        self.ax.imshow(img, cmap='viridis')
        self.ax.axis('off')
        self.ax.set_title("Attention Heatmap")
        self.canvas.draw()


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("800x500")
        self.root.resizable(True, True)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧框架（绘图区和控制按钮）
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 绘图区
        self.canvas = DigitCanvas(self.left_frame)
        self.canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 控制按钮框架
        control_frame = ttk.Frame(self.left_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # 清除按钮
        self.clear_btn = ttk.Button(control_frame, text="Clear", command=self.canvas.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 模型选择
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="CNN")
        self.model_selector = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                           values=["CNN", "Attention CNN"], state="readonly", width=15)
        self.model_selector.pack(side=tk.LEFT, padx=5)
        
        # 预测按钮
        self.predict_btn = ttk.Button(control_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建右侧框架（结果显示）
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 结果标签
        self.result_label = ttk.Label(self.right_frame, 
                                      text="Draw a digit on the left canvas, then click 'Predict'",
                                      wraplength=250, justify=tk.CENTER)
        self.result_label.pack(pady=10)
        
        # 可视化区域
        self.visualizer = PredictionVisualizer(self.right_frame)
        self.visualizer.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 初始化模型
        self.init_models()
    
    def init_models(self):
        # 检查模型文件是否存在
        self.model_paths = {
            "CNN": "models/mnist_cnn.pt",
            "Attention CNN": "models/mnist_attention.pt"
        }
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型字典
        self.models = {}
        
        # 检查是否存在预训练模型，如果不存在则提示用户
        model_exists = False
        for model_type, path in self.model_paths.items():
            if os.path.exists(path):
                model_exists = True
                break
        
        if not model_exists:
            messagebox.showwarning(
                "Model Not Found", 
                "Pre-trained model files not found. Please run train.py first to train models."
            )
    
    def load_selected_model(self):
        model_type = self.model_var.get()
        
        if model_type not in self.models:
            try:
                model_path = self.model_paths[model_type]
                
                if not os.path.exists(model_path):
                    messagebox.showwarning(
                        "Model Not Found", 
                        f"{model_type} model file not found: {model_path}. Please train the model first."
                    )
                    return None
                
                # 根据选择创建模型
                if model_type == "CNN":
                    model = DigitCNN()
                else:
                    model = AttentionDigitCNN()
                
                # 加载模型权重
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                self.models[model_type] = model
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                return None
        
        return self.models.get(model_type)
    
    def predict_digit(self):
        # 获取画布图像
        image = self.canvas.get_image()
        
        # 检查是否有绘制内容
        if np.mean(255 - image) < 5:  # 如果画布几乎是空白的
            messagebox.showinfo("Tip", "Please draw a digit on the canvas first")
            return
        
        # 加载选中的模型
        model = self.load_selected_model()
        if model is None:
            return
        
        try:
            # 预处理图像
            image_tensor = preprocess_cv_image(image)
            
            # 预测
            digit, probability = get_prediction(model, image_tensor, self.device)
            
            # 更新结果标签
            self.result_label.configure(text=f"Prediction: {digit} (Confidence: {probability:.4f})")
            
            # 更新可视化
            self.visualizer.show_prediction(digit, probability)
            
            # 如果是注意力模型，显示热力图
            model_type = self.model_var.get()
            if model_type == "Attention CNN":
                attention_map = draw_attention_heatmap(model, image_tensor, self.device)
                if attention_map is not None:
                    self.visualizer.show_image(attention_map)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")


def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 