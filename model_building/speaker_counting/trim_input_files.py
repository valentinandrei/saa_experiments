# Inputs
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/x_train_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/y_train.txt'

# Outputs
x_output =  'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/x_train_normalized_400K.txt'
y_output = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/y_train_400K.txt'

# Number of needed samples
n_samples = 300

def main():

    f_x_in = open(x_filename)
    f_y_in = open(y_filename)
    f_x_out = open(x_output, "w")
    f_y_out = open(y_output, "w")

    for i in range(n_samples):
        x_line = f_x_in.readline()
        y_line = f_y_in.readline()
        f_x_out.writelines(x_line)
        f_y_out.writelines(y_line)

    f_x_in.close()
    f_y_in.close()
    f_x_out.close()
    f_y_out.close()

if __name__ == '__main__':

    main()