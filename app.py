from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
from serpapi import GoogleSearch
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_KEY = os.getenv("SERPAPI_KEY", "26a42832f68a3fb2c572dd2ed728bc0cefcbab28791920814c40a443f4d95bd2")

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

def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    return loader(image).unsqueeze(0).to(device, torch.float)

# Neural Style Transfer Classes and Functions
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
    return torch.mm(features, features.t()).div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

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

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
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
        else:
            continue
            
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
    
    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)
    return input_img.clamp(0, 1)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'content' not in request.files:
            return redirect(request.url)
        file = request.files['content']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save content image
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
            file.save(content_path)
            
            # Process style prompt
            style_prompt = request.form.get('prompt')
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
            download_first_google_image(style_prompt, style_path)
            
            # Process images
            imsize = 512 if torch.cuda.is_available() else 256
            content_img = image_loader(content_path, imsize)
            style_img = image_loader(style_path, imsize)
            input_img = content_img.clone()
            
            # Run style transfer
            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                      content_img, style_img, input_img)
            
            # Save result
            output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
            output_image.save(os.path.join(app.config['RESULTS_FOLDER'], 'styled_image.jpg'))
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
