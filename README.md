# NeuralStyle

## Overview
Web application for neural style transfer using PyTorch, allowing users to apply artistic styles to images.

## Features
- Upload content image
- Generate style from text prompt
- Apply neural style transfer
- GPU acceleration support

## Prerequisites
- Python 3.8+
- PyTorch
- Flask
- SerpAPI key

## Installation
1. Clone repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set SerpAPI key:
```bash
export SERPAPI_KEY=your_api_key
```

## Running Application
```bash
python app.py
```
Access at `http://localhost:5000`

## Technologies
- PyTorch
- Flask
- VGG19 neural network
- Google Image Search API

## Limitations
- Requires active internet connection
- Style transfer can be computationally intensive
- Image size limited to 16MB

## License
MIT License
