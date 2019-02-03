s_dir_norep = "E:/1_Proiecte_Curente/1_Speaker_Counting/saa_experiments/paper_writing/Perception_Experiment/no_replay";
s_dir_wrep = "E:/1_Proiecte_Curente/1_Speaker_Counting/saa_experiments/paper_writing/Perception_Experiment/with_replay";

# Extract no replay files
[n_volunteers, m_individual_cat_acc, m_aggregated_cat_acc, m_conf] ...
  = get_perception_stats (s_dir_norep);

n_volunteers
m_aggregated_cat_acc
figure(1);
plot(sort(m_individual_cat_acc)); grid;
xlabel("Volunteer ID"); ylabel ("Categorical accuracy");

# Extract with replay files
[n_volunteers, m_individual_cat_acc, m_aggregated_cat_acc, m_conf] ...
  = get_perception_stats (s_dir_wrep);

n_volunteers
m_aggregated_cat_acc
figure(2);
plot(sort(m_individual_cat_acc)); grid;
xlabel("Volunteer ID"); ylabel ("Categorical accuracy");