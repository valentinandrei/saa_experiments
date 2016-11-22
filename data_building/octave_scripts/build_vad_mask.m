# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [speech, frame_start, frame_stop, activity] = ...
  build_vad_mask(wavfile, target_fs, frame_ms, sample_bits, ...
    frame_inc_ms, n_seconds)
  
  % Usage: [signal, start, stop, activity] = 
  %        build_vad_mask(wavfile, fs, ms, bits)
  %
  % The function reads a wavfile, runs the voice activity detection and then
  % returns the signal back, with two vectors containing the start and stop of
  % each frame and another vector with the voice activity duration (0 to 1).
  %
  % Input:
  %
  % wavfile       - name of .wav file in the load path
  % target_fs     - targeted sampling frequency in Hz
  % frame_ms      - the number of milliseconds per frame (multiple of 20 ms)
  % sample_bits   - targeted number of bits per sample
  % frame_inc_ms  - the increment per frame in milliseconds
  % n_seconds     - the number of seconds to be analyzed
  %
  % Output:
  %
  % speech        - 1D array containing the speech signal samples
  % frame_start   - 1D array containing the start of each frame
  % frame_stop    - 1D array containing the stop of each frame
  % activity      - 1D array with the amount of speech activity per frame (0, 1)
  
  addpath ("/home/valentin/Working/sw_tools/voicebox");
  
  debug = 0;
  
  [si, fs, n] = wavread(wavfile);
  
  if (fs ~= target_fs)
    disp 'Unaccepted sampling frequency.';
    return;
  end
   
  if (n ~= sample_bits)
    disp 'Unaccepted number of bits per sample.';
    return;
  end
  
  if (n_seconds > floor(length(si(:, 1)) / fs))
    disp 'Insufficient number of seconds in wavfile: ';
    disp wavfile;
  end
  
  speech = si(1 : n_seconds * fs, 1);
  n_samples_per_frame = fs / 1000 * frame_ms;
  n_samples_per_increment = fs / 1000 * frame_inc_ms;
  vad_decision = vadsohn(si, fs);
  n_frames = 1 + floor((n_seconds * 1000 - (frame_ms - frame_inc_ms)) / frame_inc_ms);
  frame_start = zeros(n_frames, 1);
  frame_stop = zeros(n_frames, 1);
  activity = zeros(n_frames, 1);
  
  frame_start(1) = 1;
  frame_stop(1) = n_samples_per_frame;
  activity(1) = sum(vad_decision(frame_start(1) : frame_stop(1))) / n_samples_per_frame;
  
  for i = 2 : n_frames
    
    frame_start(i) = frame_start(i-1) + n_samples_per_increment;
    frame_stop(i) = frame_start(i) + n_samples_per_frame - 1;
    activity(i) = sum(vad_decision(frame_start(i) : frame_stop(i))) / n_samples_per_frame;
  
  end
  
  if (debug == 1)
  
    subplot(3, 1, 1); plot(speech); grid;
    subplot(3, 1, 2); plot(vad_decision); grid;
    subplot(3, 1, 3); plot(activity); grid;
  
  end
  
  endfunction