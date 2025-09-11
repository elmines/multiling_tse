#!/bin/bash

store_dir=data/multiling/raw/ 
if [ ! -e $store_dir ] 
then
    mkdir $store_dir
fi
cd $store_dir

# zh, NLPCC 2016
wget -O zh_nlpcc.tsv http://tcci.ccf.org.cn/conference/2016/dldoc/evasampledata4-TaskAA.txt
# ca, Catalonian Independence
wget -O ca_independence_train.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_test.csv
wget -O ca_independence_val.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_train.csv
wget -O ca_independence_test.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_val.csv

# es, Catalonian Independence
wget -O es_independence_train.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_test.csv
wget -O es_independence_val.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_train.csv
wget -O es_independence_test.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_val.csv

# hi, Demonetisation
wget -O hi_demonetisation_tweets.txt https://raw.githubusercontent.com/sahilswami96/StanceDetection_CodeMixed/refs/heads/master/Dataset/Notebandi_tweets.txt
wget -O hi_demonetisation_stance.txt https://raw.githubusercontent.com/sahilswami96/StanceDetection_CodeMixed/refs/heads/master/Dataset/Notebandi_tweets_stance.txt
# et, Immigration
wget -O et_immigration.csv https://raw.githubusercontent.com/markmets/immigration-prediction-EST/refs/heads/main/Annotated_Dataset.csv
# fr, Macron and Lepen
wget -O fr_lepen.csv https://raw.githubusercontent.com/mirkolai/MultilingualStanceDetection/refs/heads/master/dataset/lepen_fr.csv
wget -O fr_macron.csv https://raw.githubusercontent.com/mirkolai/MultilingualStanceDetection/refs/heads/master/dataset/macron_fr.csv
# it, 2016 Constitution Referendum
wget -O it_ref.csv https://raw.githubusercontent.com/mirkolai/MultilingualStanceDetection/refs/heads/master/dataset/referendum_it.csv
