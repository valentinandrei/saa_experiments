# Unit Tests

s_file = '../wavfiles/S2.wav'
[x, FS, NBITS] = wavread(s_file);

# Test: build_vad_mask
frame_ms      = 250;
frame_inc_ms  = 100;
n_seconds     = 150;

for i = 1 : 9

  s_file = sprintf('../wavfiles/S%d.wav', i);
  
  [speech, frame_start, frame_stop, activity] = build_vad_mask(s_file, FS, ...
  frame_ms, NBITS, frame_inc_ms, n_seconds);
  
  s1 = sprintf('Number of full speech frames for file %d: %d.', i, length(find(activity == 1)));
  s2 = sprintf('Number of full silence frames for file %d: %d.', i, length(find(activity == 0)));
  
  disp(s1);
  disp(s2);
  
end