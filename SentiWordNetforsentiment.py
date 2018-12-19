#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import codes

def ReadFileSentiWordNet(filename):
    senti_word = []
    senti_pos = []
    senti_neg = []
    file = codes.open(filename,'r','utf-8')
    full_data = file.read().splitlines()
    for i in range(len(full_data)): # Với mỗi dòng trong sentiwordnet
        columns = full_data[i].split('\t')

        words = columns[4].split(' ')
        # Xét mỗi từ
        for i in range(len(words)):
            # Bỏ 2 ký tự cuối
            words[i] = words[i][:-2]
            # Xét coi có trong senti_word chưa, nếu chưa có thêm vào
            if (words[i] not in senti_word):
                senti_word.append(words[i])
                senti_pos.append(float(columns[2]))
                senti_neg.append(float(columns[3]))
    return senti_word,senti_pos,senti_neg

if __name__ == "__main__":
    # Load file SentiWordNet lên
    print "Loading SentiWordNet"
    senti_words, senti_pos, senti_neg = ReadFileSentiWordNet('SentiWordNet_3.0.0.txt')
    with open('sentiwordnet_en.txt', 'w') as f:
	 for w in senti_words:
	     f.write('%s\n' % w.encode('utf-8'))

