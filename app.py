from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)

# Set the upload folder path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Stable Diffusion model and LoRA weights
model_path = "sd-legacy/stable-diffusion-v1-5"  # Model path
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.load_lora_weights("maria26/Floor_Plan_LoRA")  # Load LoRA weights

def allowed_file(filename):
    """Check if the uploaded file has an allowed format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_floor_plan(prompt):
    """Generate a floor plan based on the prompt."""
    # Use the LoRA fine-tuned model to generate the floor plan
    image = pipe(prompt).images[0]
    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if a file is present and has the correct format
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate the floor plan using the prompt
            prompt = "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, many windows."
            generated_image = generate_floor_plan(prompt)
            generated_image.save('generated_floor_plan.png')

            return render_template("index.html", image="generated_floor_plan.png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
