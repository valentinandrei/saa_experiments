addpath("E:/1_Proiecte_Curente/1_Speaker_Counting/3rdparty/voicebox");
pkg load signal
pkg load ltfat

n_miliseconds = 100;

s_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean/84/121123/84-121123-0002.flac';
[x_signal, FS] = audioread(s_filename);

# Extract frame
n_samples = FS/1000 * n_miliseconds;
n_start = randi(length(x_signal) - n_samples);
x_frame = x_signal(n_start : n_start + n_samples);

# Envelope
x_envelope = get_speech_envelope (x_frame, FS, FS/8);
length(x_envelope)

# Spectrum
x_fft = get_speech_spectrum (x_frame, FS, FS/4);
length(x_fft)

# Histogram
x_hist =  get_histogram (x_frame, 0, 0.5, 100);
length(x_hist)

# Spectrogram
[x_specgram, v_f, v_t] = get_speech_spectrogram (x_frame, FS);

# MFCC
x_mfcc = melcepst(x_frame, FS, 'E0');
size(x_mfcc)
figure();
plot(x_mfcc);

figure();
subplot(4,1,1); plot(x_frame); grid;
subplot(4,1,2); plot(x_envelope); grid;
subplot(4,1,3); plot(x_fft); grid;
subplot(4,1,4); plot(x_hist); grid;

