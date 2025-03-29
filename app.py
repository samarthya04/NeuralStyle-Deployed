from flask import Flask, request, render_template, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from serpapi import GoogleSearch
import os
import time
import tracemalloc
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import uuid

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SerpAPI Key (replace with your actual key)
API_KEY = "26a42832f68a3fb2c572dd2ed728bc0cefcbab28791920814c40a443f4d95bd2"

def download_first_google_image(query, save_path):
    params = {
        "q": query,
        "tbm": "isch",
        "num": 1,
        "api_key": API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    if "images_results" in results and results["images_results"]:
        image_url = results["images_results"][0]["original"]
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image downloaded successfully: {save_path}")
            return True
        else:
            print("Failed to download image.")
            return False
    else:
        print("No images found.")
        return False

def image_loader(image_path, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return loader(image).unsqueeze(0).to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Load VGG19 model
cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=100000, content_weight=10):
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            if run[0]%50 == 0:
                print(f'\nRun = {run[0]}: Total Loss = {loss}')
            run[0] += 1
            return loss
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

# Attempt to import EMD
emd_installed = False
try:
    from PyEMD import EMD
    emd_installed = True
    print("EMD from PyEMD imported successfully.")
except ImportError:
    try:
        from EMD_signal import EMD
        emd_installed = True
        print("EMD from EMD_signal imported successfully.")
    except ImportError:
        print("EMD library not found. STI will use histogram comparison.")
        emd_installed = False

def calculate_sti(style_img, output):
    global emd_installed

    if emd_installed:
        try:
            emd = EMD()
            style_imfs = emd(style_img.view(-1).detach().cpu().numpy())
            output_imfs = emd(output.view(-1).detach().cpu().numpy())
            return np.mean(np.abs(style_imfs - output_imfs))
        except Exception as e:
            print(f"Error calculating STI with EMD: {e}. Falling back to histogram comparison.")
            emd_installed = False
    
    # Fallback if EMD is not available
    style_hist, _ = np.histogram(style_img.view(-1).detach().cpu().numpy(), bins=50, density=True)
    output_hist, _ = np.histogram(output.view(-1).detach().cpu().numpy(), bins=50, density=True)
    return np.mean(np.abs(style_hist - output_hist))

def calculate_content_preservation(content_img, styled_img):
    content_np = content_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    styled_np = styled_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(content_np, styled_np, multichannel=True, win_size=3, data_range=1.0)

def calculate_style_similarity(style_img, styled_img):
    style_np = style_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    styled_np = styled_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(style_np, styled_np, multichannel=True, win_size=3, data_range=1.0)

def save_output_image(output_tensor, file_path):
    output_image = output_tensor.squeeze(0).cpu().clone()
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(file_path)
    return file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Check if content image was uploaded
    if 'content_image' not in request.files:
        return jsonify({"error": "No content image uploaded"}), 400
    
    content_file = request.files['content_image']
    if content_file.filename == '':
        return jsonify({"error": "No content image selected"}), 400
    
    # Create unique session ID
    session_id = str(uuid.uuid4())
    
    # Get style prompt
    style_prompt = request.form.get('style_prompt', '')
    if not style_prompt:
        return jsonify({"error": "No style prompt provided"}), 400
    
    # Get parameters
    try:
        num_steps = int(request.form.get('num_steps', 500))
        style_weight = int(request.form.get('style_weight', 100000))
        content_weight = int(request.form.get('content_weight', 10))
    except ValueError:
        return jsonify({"error": "Invalid parameter values"}), 400
    
    # Save content image
    content_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_content.jpg")
    content_file.save(content_path)
    
    # Download style image
    style_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_style.jpg")
    if not download_first_google_image(style_prompt, style_path):
        return jsonify({"error": "Failed to download style image"}), 500
    
    # Set image size based on available resources
    imsize = 512 if torch.cuda.is_available() else 256
    
    # Load images
    try:
        content_img = image_loader(content_path, imsize)
        style_img = image_loader(style_path, imsize)
        input_img = content_img.clone()
    except Exception as e:
        return jsonify({"error": f"Error loading images: {str(e)}"}), 500
    
    # Start performance measurements
    start_time = time.time()
    tracemalloc.start()
    
    # Run style transfer
    try:
        output = run_style_transfer(
            cnn, 
            cnn_normalization_mean, 
            cnn_normalization_std, 
            content_img, 
            style_img, 
            input_img, 
            num_steps=num_steps, 
            style_weight=style_weight, 
            content_weight=content_weight
        )
    except Exception as e:
        tracemalloc.stop()
        return jsonify({"error": f"Style transfer failed: {str(e)}"}), 500
    
    # End performance measurements
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate performance metrics
    processing_time = end_time - start_time
    
    # Calculate additional metrics
    sti_score = float(calculate_sti(style_img, output))
    content_preservation_score = float(calculate_content_preservation(content_img, output))
    style_similarity_score = float(calculate_style_similarity(style_img, output))
    
    # Save output image
    output_path = os.path.join(OUTPUT_FOLDER, f"{session_id}_output.jpg")
    save_output_image(output, output_path)
    
    # Save content and style images to static folder for display
    content_display_path = os.path.join(OUTPUT_FOLDER, f"{session_id}_content_display.jpg")
    style_display_path = os.path.join(OUTPUT_FOLDER, f"{session_id}_style_display.jpg")
    save_output_image(content_img, content_display_path)
    save_output_image(style_img, style_display_path)
    
    # Return results
    result = {
        "output_image": f"/static/outputs/{session_id}_output.jpg",
        "content_image": f"/static/outputs/{session_id}_content_display.jpg",
        "style_image": f"/static/outputs/{session_id}_style_display.jpg",
        "metrics": {
            "processing_time": f"{processing_time:.2f}",
            "memory_usage": f"{peak / 10**6:.2f}",
            "sti_score": f"{sti_score:.4f}",
            "content_preservation": f"{content_preservation_score:.4f}",
            "style_similarity": f"{style_similarity_score:.4f}"
        },
        "parameters": {
            "num_steps": num_steps,
            "style_weight": style_weight,
            "content_weight": content_weight
        }
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
