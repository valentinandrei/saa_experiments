% Features' Mean Normalization and Scaling
if (b_train == 1)

  [m_features_norm, mu, sigma] = do_feature_normalization(m_features);
  save("-ascii", "x_train_normalized.txt", "m_features_norm");
  save("-ascii", "y_train.txt", "v_labels");
  
  figure();
  
  subplot(2,2,1); plot(m_features(randi(n_samples), :)); grid;
  subplot(2,2,2); plot(m_features(randi(n_samples), :)); grid;
  subplot(2,2,3); plot(m_features(randi(n_samples), :)); grid;
  subplot(2,2,4); plot(m_features(randi(n_samples), :)); grid;
  
  figure();
  
  subplot(2,2,1); plot(m_features_norm(randi(n_samples), :)); grid;
  subplot(2,2,2); plot(m_features_norm(randi(n_samples), :)); grid;
  subplot(2,2,3); plot(m_features_norm(randi(n_samples), :)); grid;
  subplot(2,2,4); plot(m_features_norm(randi(n_samples), :)); grid;  
  
else

  [m_features_norm, mu, sigma] = do_feature_normalization(m_features);
  save("-ascii", "x_test_normalized.txt", "m_features");
  save("-ascii", "y_test.txt", "v_labels");  
  
end