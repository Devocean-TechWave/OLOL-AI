from flask import Flask, jsonify, request, send_file
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import os

# Flask 앱 초기화
app = Flask(__name__)

# Stable Diffusion 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./fine_tuned_lora"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe.to(device)
pipe.safety_checker = None  

# 생성된 이미지 저장 디렉토리
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# RemoveBG API 키
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY", "your_removebg_api_key") 


def remove_background(input_path, output_path):
    """
    Remove background using RemoveBG API.
    """
    url = "https://api.remove.bg/v1.0/removebg"
    with open(input_path, "rb") as image_file:
        response = requests.post(
            url,
            files={"image_file": image_file},
            data={"size": "auto"},
            headers={"X-Api-Key": REMOVE_BG_API_KEY},
        )
    if response.status_code == 200:
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)
    else:
        raise Exception(f"RemoveBG API error: {response.status_code}, {response.text}")


@app.route('/generate-composite', methods=['POST'])
def generate_composite():
    try:
        
        data = request.form
        prompt = data.get('prompt', None)
        if not prompt or 'file' not in request.files:
            return jsonify({"error": "Prompt and file are required"}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"error": "No selected file"}), 400


        input_path = os.path.join(output_dir, "input_image.png")
        uploaded_file.save(input_path)

        # RemoveBG로 배경 제거
        extracted_path = os.path.join(output_dir, "extracted_image.png")
        remove_background(input_path, extracted_path)

        # Stable Diffusion으로 배경 생성
        with torch.no_grad():
            generated_image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        # 배경 이미지 저장
        background_path = os.path.join(output_dir, "background_image.png")
        generated_image.save(background_path)

        # 배경과 전경 합성
        foreground = Image.open(extracted_path).convert("RGBA")
        background = Image.open(background_path).convert("RGBA")
        background = background.resize(foreground.size)  # 배경 크기 조정

        composite_image = Image.alpha_composite(background, foreground)
        composite_path = os.path.join(output_dir, "composite_image.png")
        composite_image.save(composite_path, "PNG")

        return send_file(composite_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask 앱 실행
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
