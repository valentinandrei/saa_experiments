# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [c_speech, v_n_frames_speech] = build_speech_input ...
    ( v_wavfiles, fs, frame_ms, frame_inc_ms)

  % ----------------------------------------------------------------------------
  % Input:
  % ----------------------------------------------------------------------------
  % v_wavfiles        - the array with the name of all the wavfiles
  % fs                - targeted sampling frequency in Hz
  % frame_ms          - the number of milliseconds per frame (multiple of 20 ms)
  % frame_inc_ms      - the increment per frame in milliseconds
  % ----------------------------------------------------------------------------
  % Output:
  % ----------------------------------------------------------------------------
  % c_speech          - cell array with all speaker speech frames per cell
  % v_n_frames_speech - speech frames count per speaker
  
  n_files           = length(v_wavfiles);
  c_speech          = {};
  v_n_frames_speech = zeros(n_files, 1);
  
  ##############################################################################
  # Sepparate Silence and Speech
  ##############################################################################
  
  s0 = time();
  
  for i = 1 : n_files
  
    [s, start, stop, act] = get_speech_vad_mask(v_wavfiles{i}, ...
      fs, frame_ms, frame_inc_ms);
      
    [m_speech, n_speech, m_silence, n_silence] = ...
      get_speech_silence_frames(s, start, stop, act);
      
    c_speech{i} = m_speech;
    v_n_frames_speech(i) = n_speech;
      
  end
  
  s1 = time();  
  printf("Processing .wav files into speech and silence: %.3f sec.\n", s1 - s0);
  fflush(stdout);

endfunction