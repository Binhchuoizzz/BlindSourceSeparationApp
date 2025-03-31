from flask import Flask, render_template, request, send_file, jsonify
import os
import uuid
from processing import process_audio, denoise_audio

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio_file']
    mode = request.form.get("mode")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_ext = os.path.splitext(file.filename)[-1]
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}{file_ext}")
    
    file.save(input_path)
    
    output_vocal = os.path.join(OUTPUT_FOLDER, f"{file_id}_vocals.wav")
    output_music = os.path.join(OUTPUT_FOLDER, f"{file_id}_music.wav")
    
    if mode == "bss":
        process_audio(input_path, output_vocal, output_music)
    elif mode == "denoise":
        denoise_audio(input_path, output_vocal, output_music)
    else:
        return jsonify({"error": "Invalid mode"}), 400

    return jsonify({"vocal": f"/download/{file_id}_vocals.wav", "music": f"/download/{file_id}_music.wav"})

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

@app.route('/record', methods=['POST'])
def record_audio():
    audio_data = request.files['recorded_audio']
    if not audio_data:
        return jsonify({"error": "No recorded audio provided"}), 400
    
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.wav")
    
    audio_data.save(input_path)
    
    return jsonify({"message": "Recording saved", "file_url": f"/download/{file_id}.wav"})

if __name__ == '__main__':
    app.run(debug=True)
