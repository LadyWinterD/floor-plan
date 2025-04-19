from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载 Stable Diffusion 模型和 LoRA 权重
model_path = "sd-legacy/stable-diffusion-v1-5"  # 模型路径
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.load_lora_weights("maria26/Floor_Plan_LoRA")  # 加载 LoRA 权重

def allowed_file(filename):
    """检查上传的文件是否为允许的格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_floor_plan(prompt):
    """根据提示生成平面图"""
    # 使用 LoRA 微调的模型生成平面图
    image = pipe(prompt).images[0]
    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 检查是否存在文件并且文件格式正确
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 使用提示生成平面图
            prompt = "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, many windows."
            generated_image = generate_floor_plan(prompt)
            generated_image.save('generated_floor_plan.png')

            return render_template("index.html", image="generated_floor_plan.png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
