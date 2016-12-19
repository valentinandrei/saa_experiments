m_mmm = load("/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/mmm_train.txt");
m_feat_unnormalized = load("/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/x_test_unnormalized.txt");

v_max   = m_mmm(1, :);
v_min   = m_mmm(2, :);
v_mean  = m_mmm(3, :);

m_feat_normalized  = (m_feat_unnormalized - v_mean) ./ (v_max - v_min);

save("-ascii", "/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/x_test_normalized.txt", "m_feat_normalized");
