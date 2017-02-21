% Load all output files

v_files_links = load("files_link.txt");
v_files_frames = load("files_frames.txt");
v_os_detection = load("y_external.txt");
v_labels = load("overlap_label.mat");

% Allocate space for final output

n_total_frames = length(v_files_links);
n_files = length(v_files_frames);
v_results = zeros(n_files, 3);
v_results(:, 1) = v_files_frames;

% Fill final output

for i = 1 : n_total_frames - 1

  idx_file = v_files_links(i);
  b_overlap = v_os_detection(i);
  
  if (b_overlap == 0)
    v_results(idx_file, 2) += 1;
  else
    v_results(idx_file, 3) += 1;
  end  

end

save("-ascii", "ann_output.txt", "v_results");

% Run a basic accuracy test

f_no_overlap = 0.1;
f_premature = 0.3;
f_turn_stealing = 0.5;
n_correct = 0;
m_predictions = zeros(n_files, 2);

for i = 1 : n_files
  
  f_overlap_degree = v_results(i, 3) / v_results(i, 1);
  n_predicted = 0;
  
  if (f_overlap_degree < f_no_overlap)
    n_predicted = 0;
  else
    if (f_overlap_degree < f_premature)
      n_predicted = 1;
    else
      if (f_overlap_degree < f_turn_stealing)
        n_predicted = 2;
      else
        n_predicted = 3;
      end
    end
  end    
  
  if (n_predicted == v_labels.overlap_label(i))
    n_correct += 1;
  end
  
  m_predictions(i, 1) = v_labels.overlap_label(i);
  m_predictions(i, 2) = n_predicted;
  
end

printf("Accuracy: %.2f ...\n", n_correct / n_files);

v_correct = zeros(4, 1);
v_total = zeros(4, 1);

for i = 1 : n_files
  
  v_total(m_predictions(i, 1) + 1) += 1;
  
  if (m_predictions(i, 1) == m_predictions(i, 2))
    v_correct(m_predictions(i, 1) + 1) += 1;
  end    
  
end

printf("Individual accuracies : ...\n");
disp(v_correct ./ v_total);