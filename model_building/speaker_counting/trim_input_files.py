# Inputs
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/x_train_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/y_train.txt'

# Outputs
x_output =  'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/x_train_normalized_400K.txt'
y_output = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/100ms_specgram_env_hist_40s/y_train_400K.txt'

# Number of needed samples
n_samples = 350000
sz_hint = 128*1024*1024

def main():

    f_x_in = open(x_filename)
    f_y_in = open(y_filename)
    f_x_out = open(x_output, "w")
    f_y_out = open(y_output, "w")

    n_lines_read = 0
    while (n_lines_read < n_samples):
        x_lines = f_x_in.readlines(sz_hint)
        n_lines_read += len(x_lines)
        f_x_out.writelines(x_lines)

    print("Number of samples transferred " + str(n_lines_read))

    for i in range(n_lines_read):
        y_line = f_y_in.readline()
        f_y_out.writelines(y_line)

    f_x_in.close()
    f_y_in.close()
    f_x_out.close()
    f_y_out.close()

if __name__ == '__main__':

    main()