#!/bin/bash

store_dir=$(dirname $0)/../data/multiling/raw/ 
if [ ! -e $store_dir ] 
then
    mkdir $store_dir
fi
cd $store_dir


curl -L -r 0-$((2**20))  -o et_unrelated.jsonl 'https://huggingface.co/datasets/siimh/estonian_corpus_2021/resolve/main/corpus_et_clean.jsonl'
for lang in ca es fr it
do
    outpath=${lang}_globalvoices.txt.gz
    curl -L https://object.pouta.csc.fi/OPUS-GlobalVoices/v2015/mono/${lang}.txt.gz -o $outpath
    gunzip $outpath
done

# zh, C-Stance (for Unrelated data)
wget -O zh_cstance.csv https://github.com/chenyez/C-STANCE/raw/refs/heads/main/c_stance_dataset/subtaskA/raw_train_all_onecol.csv

# it, Sardinia Referendum
wget -O temp.zip https://github.com/mirkolai/evalita-sardistance/raw/refs/heads/master/sardistance-encrypted.zip
# Password prompt
unzip temp.zip development/TRAIN.csv && mv development/TRAIN.csv it_sardinia_train.csv && rm -r development
rm -r temp.zip
wget -O it_sardinia_test_labels.csv https://github.com/mirkolai/evalita-sardistance/raw/refs/heads/master/gold/TEST-GOLD.csv
wget -O temp.zip https://github.com/mirkolai/evalita-sardistance/raw/refs/heads/master/data/TEST.zip
# Password prompt
unzip temp.zip
mv TEST.csv it_sardinia_test.csv
rm temp.zip

# cs, Smoking & President Zeman dataset
curl -L https://corpora.kiv.zcu.cz/sentiment/CzechStanceDetection-v2.0.zip -o temp.zip
unzip -o temp.zip 'smoking-gold.*' 'zeman.*' 
rm temp.zip
# zh, NLPCC 2016
wget -O zh_nlpcc.tsv http://tcci.ccf.org.cn/conference/2016/dldoc/evasampledata4-TaskAA.txt
# # ca, Catalonian Independence
wget -O ca_independence_train.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_train.csv
wget -O ca_independence_val.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_val.csv
wget -O ca_independence_test.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/01_CIC_CA/catalan_test.csv

# es, Catalonian Independence
wget -O es_independence_train.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_train.csv
wget -O es_independence_val.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_val.csv
wget -O es_independence_test.csv https://raw.githubusercontent.com/ZotovaElena/Multilingual-Stance-Detection/refs/heads/master/CIC_2020_Dataset/data/02_CIC_ES/spanish_test.csv

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
