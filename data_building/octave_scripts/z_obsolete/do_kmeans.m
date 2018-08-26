# Author: Valentin Andrei
# E-Mail: am_valentin@yahoo.com

function [m_centroids] = do_kmeans (f_features, f_labels, n_samples, n_clusters, f_epsilon, n_conv)

  # Read input data
  
  printf("Reading features file: ...\n");
  fflush(stdout);
  m_features = load(f_features);
  m_features = m_features(1 : n_samples, :); 
  printf("Reading labels file: ...\n");
  fflush(stdout);
  m_labels = load(f_labels);
  m_labels = m_labels(1 : n_samples, :);
  [v_temp, v_labels] = max(m_labels');
  n_features = size(m_features)(2);
  
  # Smart allocation of centroids
  printf("Picking centroids because we actually know the labels: ...\n");
  fflush(stdout);
  m_centroids = zeros(n_clusters, n_features);
  v_centroids_followers = zeros(n_clusters, 1);
  for i = 1 : n_samples
    n_class = v_labels(i);
    m_centroids(n_class, :) += m_features(i, :);
    v_centroids_followers(n_class) += 1;
  end
  m_centroids = m_centroids ./ v_centroids_followers;
  
  # Run k-means algorithm
  printf("Running k-means: ...\n");
  t0 = time();
  fflush(stdout);
  n_iteration = 1;
  n_current_conv = 0;  
  m_new_centroids = ones(n_clusters, n_features);
  v_following = zeros(n_samples, 1);
  
  while ((n_current_conv < n_conv) &&(n_iteration < 200))
    
    # Reset the new centroids container
    m_new_centroids = zeros(n_clusters, n_features);
    v_centroids_followers = zeros(n_clusters, 1);
    
    # Clustering phase
    printf("iteration %d ...\n", n_iteration);
    fflush(stdout);
    for i = 1 : n_samples
      v_distances = sum(sqrt(abs(m_features(i, :) .^ 2 - m_centroids .^ 2)), 2);
      [f_dist, n_centroid] = min(v_distances);
      v_centroids_followers(n_centroid) += 1;
      m_new_centroids(n_centroid, :) += m_features(i, :);
      v_following(i) = n_centroid; 
    end
    
    # If a centroid does not have any attached followers
    for i = 1 : n_clusters
      if (v_centroids_followers(i) == 0)
        m_new_centroids(i, :) = m_centroids(i, :);
        v_centroids_followers(i) = 1;
      end
    end
    
    # Allocate new centroids
    m_new_centroids = m_new_centroids ./ v_centroids_followers;
    f_delta = max(sum(abs(m_centroids .^ 2 - m_new_centroids .^ 2), 2));  
    m_centroids = m_new_centroids;
    
    # Verify convergence
    printf("f_delta = %.6f ...\n", f_delta); 
    disp(v_centroids_followers'); fflush(stdout);
    if (f_delta < f_epsilon)
      n_current_conv += 1;
    else
      n_current_conv = 0;
    end
    
    n_iteration += 1;
    
  end
  
  t1 = time();
  printf("Elapsed time: %.2f seconds ...\n", t1 - t0); fflush(stdout);
  
  # Compute classification error rates using assumed centroids
  m_classification = zeros(n_clusters, 2);
  for i = 1 : n_samples
    if (v_labels(i) == v_following(i))
      m_classification(v_labels(i), 1) += 1;
    else
      m_classification(v_labels(i), 2) += 1;
    end
  end
  
  # Print classification errors
  m_ratios = m_classification ./ sum(m_classification, 2) * 100;
  v_ratios_totals = sum(m_ratios);
  disp(m_ratios);
  printf("Correct classification ratio: %.3f pc ...\n", v_ratios_totals(1) / ...
    sum(v_ratios_totals)); fflush(stdout);  
  
  # Display results
  printf("Each centroid followers: ...\n");
  disp(v_centroids_followers); fflush(stdout);  
  
  v_distances = zeros(n_samples, 1);
  for i = 1 : n_samples
    n_centroid = v_following(i);
    f_distance = sum(abs(m_features(i, :) .^ 2 - m_centroids(n_centroid, :) .^ 2));
    v_distances(i) = n_centroid + f_distance;    
  end
  
  hist(v_distances);
  colormap(summer());  
  
endfunction