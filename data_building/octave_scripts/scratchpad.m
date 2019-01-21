% Transform one hot vector
[ix, iy] = find(v_labels_oh);
v_labels = iy;

% Run printing
b_example_available = 1;
while (b_example_available == 1)
  
  figure();
  
  n_speakers = 0;
  
  for i = 1 : n_max_speakers    
    for j = 1 : n_set_size      
      if (v_labels(j) == i)        
        subplot(2, 4, i);
        [S, v_f, v_t] = get_speech_spectrogram (m_mixtures(j, :), fs);
        imagesc(v_t, v_f, log(S));
        set (gca, "ydir", "normal");
        
        subplot(2, 4, i + 4);
        [s_envelope] = get_speech_envelope(m_mixtures(j, :), fs, fs / 128);
        plot(s_envelope); grid;
        
        n_speakers += 1
        v_labels(j) = 0;
 
        break; 
      end
    endfor
  endfor
  
  if (n_speakers == n_max_speakers)
    b_example_available = 1;
  else
    b_example_available = 0;
  end

endwhile

