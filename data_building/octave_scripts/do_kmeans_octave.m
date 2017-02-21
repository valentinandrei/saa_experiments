## Author: valentin <valentin@valentin-laptop>
## Created: 2017-02-03

function [m_centroids, m_assumed_centroids] = do_kmeans_octave (f_features, f_labels, n_clusters, n_samples)

  pkg load statistics;

  # Read input data
  printf("Reading input files: ...\n"); fflush(stdout);
  t0 = time();
  m_features = load(f_features);
  m_features = m_features(1 : n_samples, :);
  n_features = size(m_features, 2);
  m_labels = load(f_labels);
  m_labels = m_labels(1 : n_samples, :);
  [v_temp, v_labels] = max(m_labels');
  t1 = time();
  printf("Elapsed time: %.3f seconds ...\n", t1 - t0); fflush(stdout); 
  
  # Smart allocation of centroids
  printf("Assuming centroids because we actually know the labels: ...\n");
  fflush(stdout);
  m_assumed_centroids = zeros(n_clusters, n_features);
  v_centroids_followers = zeros(n_clusters, 1);
  for i = 1 : n_samples
    n_class = v_labels(i);
    m_assumed_centroids(n_class, :) += m_features(i, :);
    v_centroids_followers(n_class) += 1;
  end
  m_assumed_centroids = m_assumed_centroids ./ v_centroids_followers;
  
  # Run k-means
  printf("Running k-means: ...\n");
  fflush(stdout);
  t0 = time();
  [v_following, m_centroids] = kmeans(m_features, n_clusters, 'Start', ...
    m_assumed_centroids);
  t1 = time();
  printf("Elapsed time: %.3f seconds ...\n", t1 - t0); fflush(stdout);
  
  # Assume each centroid is closest to the one determined using labeled data
  for i = 1 : n_clusters
    v_centroid = m_centroids(i, :);
    v_dist_to_centroids = sum(abs(v_centroid .^ 2 - m_assumed_centroids .^ 2), 2);
    [temp, n_correspondence] = min(v_dist_to_centroids);
    v_coresponding_centroid(i) = n_correspondence;
  end
  
  printf("Corresponding centroids: ...\n");
  disp(v_coresponding_centroid); fflush(stdout);
  
  # Compute classification error rates using assumed centroids
  m_classification = zeros(n_clusters, 2);
  for i = 1 : n_samples
    if (v_labels(i) == v_following(i)) # v_coresponding_centroid(v_following(i)))
      m_classification(v_labels(i), 1) += 1;
    else
      m_classification(v_labels(i), 2) += 1;
    end    
  end
  
  # Print classification errors
  m_ratios = m_classification ./ sum(m_classification, 2) * 100;
  v_ratios_totals = sum(m_ratios);
  disp(m_ratios); fflush(stdout);  
  printf("Correct classification ratio: %.3f pc ...\n", v_ratios_totals(1) / ...
    sum(v_ratios_totals));
  
  # Display Histograms
  
  figure();
  hist(v_following, 1 : 5); grid;
  
  figure();
  v_distances = zeros(n_samples, 1);
  for i = 1 : n_samples
    n_centroid = v_following(i);
    f_distance = sum(abs(m_features(i, :) .^ 2 - m_centroids(n_centroid, :) .^ 2));
    v_distances(i) = n_centroid + f_distance;    
  end
  
  # Display histogram for each centroid follower
  subplot(2, 3, 1);
  v_bins = 0 : 0.1 : 10;
  hist(v_distances, v_bins); grid;
  colormap(summer());
  
  for i = 1 : n_clusters
    v_cluster = [];
    for j = 1 : n_samples
      if (v_following(j) == i)
        v_cluster = [v_cluster, v_distances(j)];
      end
    end
    
    subplot(2, 3, i + 1);
    hist(v_cluster, v_bins); grid;
    colormap(summer());  
  end
  
endfunction
