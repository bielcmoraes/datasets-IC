
import csv
import os
import shutil
import run_sosc
import argparse
from run_sosc import *


def divide_batch():
    '''
    Batch formation from two data sets for each fold
    '''

    input_file1 = os.path.join(data_dir, 'StackOverflow_Original.csv')

    num_of_sentences_ICSE = 0  # total number of sentences in the data set
    all_sentences_ICSE = []
    sentence_id_from_csv_ICSE = []
    neg_sen_ICSE = 178
    neu_sen_ICSE = 1191
    pos_sen_ICSE = 131

    with open(input_file1, "r") as f1:
        reader = csv.DictReader(f1)
        for row in reader:
            num_of_sentences_ICSE = num_of_sentences_ICSE + 1
            sentence_id_from_csv_ICSE.append(row["id"])
            all_sentences_ICSE.append(row["text"])

    input_file2 = os.path.join(data_dir, 'NewData.csv')

    num_of_sentences_ND = 0  # total number of sentences in the data set
    all_sentences_ND = []
    sentence_id_from_csv_ND = []
    neg_sen_ND = 1119
    neu_sen_ND = 2305
    pos_sen_ND = 576

    with open(input_file2, "r") as f2:
        reader = csv.DictReader(f2)
        for row in reader:
            num_of_sentences_ND = num_of_sentences_ND + 1
            sentence_id_from_csv_ND.append(row["id"])
            all_sentences_ND.append(row["text"])



    total_sentences = num_of_sentences_ICSE + num_of_sentences_ND
    eachbatchsize = int(total_sentences/10)
    negI_index = 0
    neuI_index = neg_sen_ICSE
    posI_index = neg_sen_ICSE + neu_sen_ICSE
    negE_index = 0
    neuE_index = neg_sen_ND
    posE_index = neg_sen_ND + neu_sen_ND
    batch_list = []
    batch_list_train = []

    for i in range(num_of_folds):
        negI = [17, 17, 18, 18, 18, 18, 18, 18, 18, 18]
        negE = [111, 112, 112, 112, 112, 112, 112, 112, 112, 112]
        posI = [13, 13, 13, 13, 13, 13, 13, 13, 13, 14]
        posE = [57, 57, 57, 57, 58, 58, 58, 58, 58, 58]
        neuI = [119, 119, 119, 119, 119, 119, 119, 119, 119, 120]
        neuE = eachbatchsize - (negI[i] + negE[i] + posI[i] + posE[i] + neuI[i])
        neuE_undersample = 330 - (negI[i] + negE[i] + posI[i] + posE[i] + neuI[i])
        temp_sentences = []
        temp_sentences_train = []

        for ni in range(negI[i]):
            temp_sentences.append([sentence_id_from_csv_ICSE[negI_index], all_sentences_ICSE[negI_index], 'Negative'])
            temp_sentences_train.append([sentence_id_from_csv_ICSE[negI_index], all_sentences_ICSE[negI_index], 'Negative'])
            negI_index = negI_index + 1

        for ne in range(negE[i]):
            temp_sentences.append([sentence_id_from_csv_ND[negE_index], all_sentences_ND[negE_index], 'Negative'])
            temp_sentences_train.append([sentence_id_from_csv_ND[negE_index], all_sentences_ND[negE_index], 'Negative'])
            negE_index = negE_index + 1

        for ui in range(neuI[i]):
            temp_sentences.append([sentence_id_from_csv_ICSE[neuI_index], all_sentences_ICSE[neuI_index], 'Neutral'])
            temp_sentences_train.append([sentence_id_from_csv_ICSE[neuI_index], all_sentences_ICSE[neuI_index], 'Neutral'])
            neuI_index = neuI_index + 1

        for ue in range(neuE):
            temp_sentences.append([sentence_id_from_csv_ND[neuE_index], all_sentences_ND[neuE_index], 'Neutral'])
            if undersample == 'yes': # for undersampling, we only reduce the number of neutral sentences from the training data
                if ue < neuE_undersample:
                    temp_sentences_train.append([sentence_id_from_csv_ND[neuE_index], all_sentences_ND[neuE_index], 'Neutral'])
            else:
                temp_sentences_train.append([sentence_id_from_csv_ND[neuE_index], all_sentences_ND[neuE_index], 'Neutral'])
            neuE_index = neuE_index + 1


        for pi in range(posI[i]):
            temp_sentences.append([sentence_id_from_csv_ICSE[posI_index], all_sentences_ICSE[posI_index], 'Positive'])
            temp_sentences_train.append([sentence_id_from_csv_ICSE[posI_index], all_sentences_ICSE[posI_index], 'Positive'])
            posI_index = posI_index + 1

        for pe in range(posE[i]):
            temp_sentences.append([sentence_id_from_csv_ND[posE_index], all_sentences_ND[posE_index], 'Positive'])
            temp_sentences_train.append([sentence_id_from_csv_ND[posE_index], all_sentences_ND[posE_index], 'Positive'])
            posE_index = posE_index + 1

        batch_list.append(temp_sentences)
        batch_list_train.append(temp_sentences_train)
    return batch_list, batch_list_train

 # divides the entire dataset into 10 batches

def create_trn_dev_test_set(data_dir, fold, num_of_folds, batches, train_batches):
    test_data = batches[fold]
    train_data = train_batches[:fold] + train_batches[fold+1:]
    with open(os.path.join(data_dir,'test.tsv'), 'w') as tst:
        tst = csv.writer(tst, delimiter='\t')
        for list_items in test_data:
            tst.writerow(list_items)

    with open(os.path.join(data_dir, 'train.tsv'), 'w') as trn:
        trn = csv.writer(trn, delimiter='\t')
        for nb in range(len(train_data)):
            for list_items in train_data[nb]:
                trn.writerow(list_items)

if __name__ == "__main__":
    ######Configuration for running BERT 10 fold ########
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='datasets', help='Directory for the dataset')
    parser.add_argument('-o', '--out_dir', default='out', help='Directory for the output')
    args = parser.parse_args()
    root_dir = os.getcwd()
    ###Temporary directory for saving BERT results###
    source_dir = os.path.join(root_dir, 'sosc_output')
    out_dir = os.path.join(root_dir, args.out_dir)
    num_of_folds = 10
    undersample = 'no' #provide yes or no
    data_dir = os.path.join(root_dir, args.data_dir)
    batches, train_batches = divide_batch()
    for fold in range(num_of_folds):
        ext = 'sosc_output_10fold' + str(fold)
        des_dir = os.path.join(out_dir, ext)
        os.mkdir(des_dir)
        create_trn_dev_test_set(data_dir, fold, num_of_folds, batches, train_batches)
        run_sosc.main()
        dest = shutil.move(source_dir, des_dir, copy_function = shutil.copytree)






