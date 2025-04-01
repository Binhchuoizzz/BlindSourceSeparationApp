import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import pywt
import noisereduce as nr

def high_pass_filter(audio, sr, cutoff=150):
    """Lọc High-Pass để loại bỏ nhiễu tần số thấp."""
    sos = signal.butter(6, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def low_pass_filter(audio, sr, cutoff=6000):
    """Lọc Low-Pass để giữ lại phần giọng hát, loại bỏ nhạc nền."""
    sos = signal.butter(6, cutoff, btype='lowpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def separate_hpss(audio, sr, margin_vocal=2.0, margin_music=1.2):
    """Tách giọng hát và nhạc nền bằng HPSS với margin tùy chỉnh."""
    stft_audio = librosa.stft(audio)
    harmonic, percussive = librosa.decompose.hpss(stft_audio, margin=(margin_vocal, margin_music))
    return librosa.istft(harmonic), librosa.istft(percussive)

def wavelet_denoise(audio, wavelet="db6", level=3):
    """Lọc nhiễu tín hiệu bằng Wavelet Transform."""
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)

def reduce_noise(audio, sr):
    """Giảm nhiễu nền bằng Spectral Gating từ thư viện noisereduce."""
    return nr.reduce_noise(y=audio, sr=sr, stationary=True)

def adaptive_noise_subtraction(clean, noise):
    """Lọc nhiễu thích ứng bằng cách trừ tín hiệu nhiễu."""
    noise = librosa.effects.time_stretch(noise, rate=1.1)  # Điều chỉnh độ dài noise
    noise = np.pad(noise, (0, max(0, len(clean) - len(noise))), mode='constant')
    return clean - noise[:len(clean)]

def denoise_audio(input_path, output_voice, output_noise):
    """Lọc nhiễu từ file âm thanh hội thoại, xuất ra file giọng nói sạch và nhiễu nền."""
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Lọc nhiễu
    reduced_noise = reduce_noise(audio, sr)
    reduced_noise = wavelet_denoise(reduced_noise)
    reduced_noise = high_pass_filter(reduced_noise, sr)

    # Lấy phần nhiễu
    noise_only = adaptive_noise_subtraction(audio, reduced_noise)

    # Lưu file kết quả
    sf.write(output_voice, reduced_noise, sr)
    sf.write(output_noise, noise_only, sr)

    return output_voice, output_noise

def normalize_audio(audio):
    """Chuẩn hóa âm lượng."""
    return audio / (np.max(np.abs(audio)) + 1e-7)

def process_audio(input_path, output_vocal, output_music, mode="separate_music"):
    """Xử lý tách giọng hát/nhạc hoặc lọc nhiễu giọng nói."""
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    if mode == "separate_music":
        # Tách giọng hát và nhạc nền
        vocal, music = separate_hpss(audio, sr)
        vocal, music = wavelet_denoise(vocal), wavelet_denoise(music)
        vocal, music = high_pass_filter(vocal, sr), low_pass_filter(music, sr)
    
    elif mode == "denoise_speech":
        # Lọc nhiễu giọng nói
        vocal = reduce_noise(audio, sr)
        vocal = wavelet_denoise(vocal)
        vocal = high_pass_filter(vocal, sr)
        music = None  # Không có nhạc nền trong chế độ này
    
    # Chuẩn hóa và lưu file
    vocal = normalize_audio(vocal)
    sf.write(output_vocal, vocal, sr)
    if music is not None:
        music = normalize_audio(music)
        sf.write(output_music, music, sr)
    
    return output_vocal, output_music if music is not None else None