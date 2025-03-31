import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import pywt
from sklearn.decomposition import FastICA

def high_pass_filter(audio, sr, cutoff=150):
    """Lọc High-Pass để loại bỏ nhiễu tần số thấp."""
    sos = signal.butter(6, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def low_pass_filter(audio, sr, cutoff=6000):
    """Lọc Low-Pass để giữ lại phần giọng hát, loại bỏ nhạc nền."""
    sos = signal.butter(6, cutoff, btype='lowpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def separate_hpss(audio, sr, n_fft=2048, hop_length=512, margin=1.5):
    """Tách giọng hát và nhạc nền bằng HPSS (cải thiện)."""
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    harmonic, percussive = librosa.decompose.hpss(stft_audio, margin=margin)
    
    vocal_audio = librosa.istft(harmonic, hop_length=hop_length)
    music_audio = librosa.istft(percussive, hop_length=hop_length)

    return vocal_audio, music_audio

def apply_ica(vocal_audio, music_audio, sr):
    """Áp dụng ICA để tối ưu phân tách giọng hát và nhạc nền."""
    X = np.c_[vocal_audio, music_audio]  # Ghép tín hiệu thành ma trận nguồn
    ica = FastICA(n_components=2, max_iter=1000)
    sources = ica.fit_transform(X)  # Tách các nguồn độc lập

    # Xác định nguồn nào là giọng hát bằng phân tích phổ tần số
    def dominant_frequency(signal, sr):
        fft_spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1/sr)
        return np.sum(freqs * fft_spectrum) / np.sum(fft_spectrum)  # Trọng số phổ

    freq1 = dominant_frequency(sources[:, 0], sr)
    freq2 = dominant_frequency(sources[:, 1], sr)

    # Giọng hát thường có tần số trọng tâm từ 300Hz - 4000Hz
    if 300 <= freq1 <= 4000:
        return sources[:, 0], sources[:, 1]  # Vocal - Music
    else:
        return sources[:, 1], sources[:, 0]  # Hoán đổi nếu cần

def wavelet_denoise(audio, wavelet="db6", level=3):
    """Lọc nhiễu tín hiệu bằng Wavelet Transform."""
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745  # Tính ngưỡng
    coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)

def normalize_audio(audio):
    """Chuẩn hóa âm lượng."""
    return audio / (np.max(np.abs(audio)) + 1e-7)

def process_audio(input_path, output_vocal, output_music):
    """Tách giọng hát và nhạc nền kết hợp HPSS + ICA + Wavelet Transform."""
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Bước 1: HPSS tách sơ bộ
    vocal, music = separate_hpss(audio, sr)

    # 🔥 Sửa lỗi: Truyền thêm `sr` vào `apply_ica()`
    vocal_ica, music_ica = apply_ica(vocal, music, sr)

    # Bước 3: Lọc tạp âm bằng Wavelet Transform
    vocal_clean = wavelet_denoise(vocal_ica)
    music_clean = wavelet_denoise(music_ica)

    # Bước 4: Lọc High-Pass và Low-Pass
    vocal_clean = high_pass_filter(vocal_clean, sr)
    vocal_clean = low_pass_filter(vocal_clean, sr)

    # Bước 5: Chuẩn hóa tín hiệu
    vocal_clean = normalize_audio(vocal_clean)
    music_clean = normalize_audio(music_clean)

    # Lưu file đầu ra
    sf.write(output_vocal, vocal_clean, sr)
    sf.write(output_music, music_clean, sr)

    return output_vocal, output_music
