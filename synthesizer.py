import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.fftpack import fft
from IPython.display import display


DATA_LIM = 5000 # limit of data from wav file

def fft_wavfile(wav_filename):
    '''
    Plot the FFT of a wav file
    :param wav_file: filename
    '''
    f_sampling, sound_data = wavfile.read(wav_filename)
    first_track = sound_data  # first track of a two track signal
    if sound_data.T.shape[0] == 2:  # 2 tracks
        first_track = sound_data.T[0]
    norm_data = [(ele/2**8)*2-1 for ele in first_track]  # 8-bit track, normalizing b on [-1,1)
    amps = abs(fft(norm_data)[1:DATA_LIM])
    k = np.arange(1,DATA_LIM) / (len(sound_data)/f_sampling)
    
    plt.plot(k, amps, 'b')
    plt.title('FFT of {}'.format(wav_filename))
    plt.xlabel('Hz')
    plt.ylabel('Amplitude')
    plt.show()
    
    freqs_amps = zip(k, amps)
    freqs_amps = sorted(freqs_amps, key=lambda f: f[1], reverse=True)  # sort by amplitude
    
    top_tones = []
    for f, amp in freqs_amps:
        # don't add new frequencies within 20 Hz
        if all([abs(abs(f)-abs(x)) > 20 for x in top_tones]):
                top_tones.append(f)
        
    print("Top 4 tones:", sorted([round(t, 2) for t in top_tones[:5]]))
    print()
    display(freqs_amps[:20])

fft_wavfile("clarinet_A3.wav")
fft_wavfile("flute_A3.wav")
fft_wavfile("guitar_A3.wav")

# done manually by looking at freqs_amps for now

G_F0 = 596045.14  # guitar fundamental
GUITAR_RATIOS = [419651.47 / G_F0, 92892.88 / G_F0, 57943.35 / G_F0]

F_F0 = 577696.68  # flute fundamental
FLUTE_RATIOS = [829705.46 / F_F0, 388099.15 / F_F0, 313152.34 / F_F0]

C_F0 = 600777.15  # clarinet fundamental
CLARINET_RATIOS = [0, 472392.84 /  C_F0, 0, 252482.36 / C_F0]


# Parameters for waveform generation
duration = 2.0  # Duration in seconds
sample_rate = 44100  # Sample rate (samples per second)
volume = 0.5  # Volume level (0.0 to 1.0)

# Generate time array
t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)

def create_mp3(freq, ratios):
    '''
    Create an MP3 file with fundamental freqeucny 'freq' and overtones ratio 'ratios'.
    :param freq: float, the fundamental frequency
    :param ratios: list of relative ratios - index i is the amplitude of ith overtone / fundamental amplitude

    :return: pydub.AudioSegment object
    '''
   
    # Generate guitar-like waveform using sine waves
    fundamental_wave = np.sin(2 * np.pi * freq * t)
    overtones = []
    for n,r in enumerate(ratios):
        overtones.append(r * np.sin(2 * np.pi * (n+2) * freq * t))  # first element is 2nd harmonic
    combined_waveform = volume * (fundamental_wave + sum(overtones))
    
    # Convert waveform to 16-bit PCM format
    pcm_waveform = (combined_waveform * (2**15 - 1)).astype(np.int16)
    
    # Create AudioSegment from waveform
    return AudioSegment(
        data=pcm_waveform.tobytes(),
        sample_width=2,  # 16-bit PCM
        frame_rate=sample_rate,
        channels=1  # Mono
    )


# create C3 to C4 mp3 files
f_A3 = 220  # Hz
note_names = ['C3', 'C3s', 'D3', 'D3s', 'E3', 'F3', 'F3s', 'G3', 'G3s', 'A3', 'A3s', 'B3',
              'C4', 'C4s', 'D4', 'D4s', 'E4', 'F4', 'F4s', 'G4', 'G4s', 'A4', 'A4s',  'B4', 'C5']

def note_to_name(n):
    return note_names[n-1]

for n in range(len(note_names)):
    # 1 is A3, 24 is B4
    freq = f_A3 * 2 ** ((n-9)/12.0)  # 9th note is A3
    print(freq, "notes/{}.mp3".format(note_names[n]))
    audio_segment = create_mp3(freq, CLARINET_RATIOS)
    audio_segment.export("VirtualPiano/sons/{}.mp3".format(note_names[n]), format="mp3")
