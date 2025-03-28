import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal

def high_pass_filter(audio, sr, cutoff=100):
    """
    Lọc High-Pass để loại bỏ tiếng ồn tần số thấp, giúp giọng hát rõ hơn.
    """
    sos = signal.butter(6, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def low_pass_filter(audio, sr, cutoff=8000):
    """
    Lọc Low-Pass giúp giữ lại phần giọng hát, loại bỏ tạp âm nhạc nền.
    """
    sos = signal.butter(6, cutoff, btype='lowpass', fs=sr, output='sos')
    return signal.sosfilt(sos, audio)

def separate_hpss(audio, sr, n_fft=4096, hop_length=1024, margin=3.0):
    """
    Tách giọng hát và nhạc nền bằng HPSS.
    """
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # HPSS - Đảo lại kết quả để giọng hát thực sự là giọng hát
    percussive, harmonic = librosa.decompose.hpss(stft_audio, margin=margin)

    # Chuyển về dạng sóng
    vocal_audio = librosa.istft(harmonic, hop_length=hop_length)
    music_audio = librosa.istft(percussive, hop_length=hop_length)

    # Lọc high-pass cho giọng hát
    vocal_audio = high_pass_filter(vocal_audio, sr)
    
    # Lọc low-pass để giữ giọng hát, tránh tạp âm
    vocal_audio = low_pass_filter(vocal_audio, sr)

    return vocal_audio, music_audio

def normalize_audio(audio):
    """
    Chuẩn hóa âm lượng để tránh quá nhỏ hoặc méo tiếng.
    """
    return audio / (np.max(np.abs(audio)) + 1e-7)

def process_audio(input_path, output_vocal, output_music):
    """
    Xử lý tách giọng hát và nhạc nền.
    """
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Tách giọng hát và nhạc nền
    vocal, music = separate_hpss(audio, sr)

    # Chuẩn hóa âm lượng
    vocal = normalize_audio(vocal)
    music = normalize_audio(music)

    # Lưu file đầu ra
    sf.write(output_vocal, vocal, sr)
    sf.write(output_music, music, sr)

    return output_vocal, output_music
