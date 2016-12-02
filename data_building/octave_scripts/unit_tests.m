# Unit Tests

s_file = '../wavfiles/S2.wav'
[x, FS, NBITS] = wavread(s_file);

# Test: build_vad_mask
frame_ms      = 200;
frame_inc_ms  = 50;

for i = 1 : 9

  s_file = sprintf('../wavfiles/S%d.wav', i);
  
  [speech, frame_start, frame_stop, activity] = build_vad_mask(s_file, FS, ...
  frame_ms, NBITS, frame_inc_ms);
  
  s1 = sprintf('Number of full speech frames for file %d: %d.', i, length(find(activity == 1)));
  s2 = sprintf('Number of full silence frames for file %d: %d.', i, length(find(activity == 0)));
  
  disp(s1);
  disp(s2);
  
  [m_speech_frames, n_speech, m_silence_frames, n_silence] = ...
    get_speech_silence_frames (speech, frame_start, frame_stop, activity);
    
  s1 = sprintf('Number of selected full speech frames for file %d: %d, %d', 
    i, length(m_speech_frames), n_speech);
  s2 = sprintf('Number of selected full silence frames for file %d: %d, %d', 
    i, length(m_silence_frames), n_silence);
    
  # disp(s1);
  # disp(s2);
  
end