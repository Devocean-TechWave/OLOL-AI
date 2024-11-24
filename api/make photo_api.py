from flask import Flask, jsonify, request, send_file
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Define Flask app
app = Flask(__name__)

# Stable Diffusion 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./fine_tuned_lora"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe.to(device)  
pipe.safety_checker = None  

# 생성된 이미지 저장 경로
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)


@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # 요청으로부터 프롬프트를 가져옴
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt not provided"}), 400

        prompt = data['prompt']
        num_steps = data.get('steps', 50) 
        guidance_scale = data.get('guidance_scale', 7.5)  

        # Stable Diffusion으로 이미지 생성
        with torch.no_grad():
            generated_image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]

        # 이미지 저장
        output_path = os.path.join(output_dir, "generated_image.png")
        generated_image.save(output_path)
        
        # 이미지 반환
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask 앱 실행
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
