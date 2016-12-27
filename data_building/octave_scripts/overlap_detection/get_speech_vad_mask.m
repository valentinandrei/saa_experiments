# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [speech, frame_start, frame_stop, activity] = ...
  get_speech_vad_mask(wavfile, target_fs, frame_ms, frame_inc_ms)
  
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
  speech = si(:, 1);
  
  if (fs ~= target_fs)
  
    printf('Resampling signal...\n');
    fflush(stdout);
    
    speech = resample(speech, target_fs, fs);
    fs = target_fs;
  end
  
  n_seconds               = floor(length(speech) / fs);
  speech                  = speech(1 : n_seconds * fs);
  n_samples_per_frame     = fs / 1000 * frame_ms;
  n_samples_per_increment = fs / 1000 * frame_inc_ms;
  
  t0 = time();
  vad_decision            = vadsohn(speech, fs);
  t1 = time();
  printf("vadsohn duration: %.3f seconds\n", t1 - t0);
  fflush(stdout);
  
  n_frames                = floor((n_seconds * 1000 - (frame_ms - frame_inc_ms)) / frame_inc_ms) - 1;
  frame_start             = zeros(n_frames, 1);
  frame_stop              = zeros(n_frames, 1);
  activity                = zeros(n_frames, 1);
  
  frame_start(1)          = 1;
  frame_stop(1)           = n_samples_per_frame;
  activity(1)             = sum(vad_decision(frame_start(1) : frame_stop(1))) / n_samples_per_frame;
  
  for i = 2 : n_frames
    
    frame_start(i)  = frame_start(i-1) + n_samples_per_increment;
    frame_stop(i)   = frame_start(i) + n_samples_per_frame - 1;
    activity(i)     = sum(vad_decision(frame_start(i) : frame_stop(i))) / n_samples_per_frame;
  
  end
  
  t2 = time();
  printf("frame start-stop creation duration: %.3f seconds\n", t2 - t1);
  fflush(stdout);
  
  if (debug == 1)
  
    figure();
    subplot(2, 1, 1); plot(si(:, 1)); grid;
    subplot(2, 1, 2); plot(speech); grid;
    
    figure();
    subplot(3, 1, 1); plot(speech); grid;
    subplot(3, 1, 2); plot(vad_decision); grid;
    subplot(3, 1, 3); plot(activity); grid;
  
  end
  
  endfunction