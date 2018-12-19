import os
import ast
import spacy
import numpy as np
import xml.etree.ElementTree as ET
from errno import ENOENT
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding('utf8')

nlp = spacy.load("en")


def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, max_sentence_len, max_aspect_len = {}, 0, 0
    word2id['<pad>'] = 0
    print('pre_processed se la: %s' % pre_processed) 
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
		#print('Xuat line ...')
		#print('line se la: %s' % line)  
                content = line.strip().split()
		#print('Xuat content tach tu bang lenh split ...')
		#print('content se la: %s' % content)
                if len(content) == 3:
                    max_sentence_len = int(content[1])
                    max_aspect_len = int(content[2])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        train_tree = ET.parse(train_fname)
        train_root = train_tree.getroot()
	
        for sentence in train_root:
	    #print('sentence se la: %s' % sentence)
            sptoks = nlp(sentence.find('text').text.decode())
	    #print('sptoks se la: %s' % sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
	    #print('words se la: %s' % words)
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    if asp_term.get('polarity') == 'conflict':
                        continue
                    t_sptoks = nlp(asp_term.get('term').decode())
		    #print('t_sptoks se la: %s' % t_sptoks)
                    if len(t_sptoks) > max_aspect_len:
                        max_aspect_len = len(t_sptoks)
        word_count = Counter(words).most_common()
	#print('word_count se la: %s' % word_count)
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)
		#print('word se la: %s' % word)
		#print('len(word2id) se la: %s' % len(word2id))
		#print('lword2id[word] se la: %s' % word2id[word])
		
	#print('het file train')
        test_tree = ET.parse(test_fname)
        test_root = test_tree.getroot()
        for sentence in test_root:
            sptoks = nlp(sentence.find('text').text.decode())
	    #print('sptoks se la: %s' % sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) > max_sentence_len:
                max_sentence_len = len(sptoks)
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    if asp_term.get('polarity') == 'conflict':
                        continue
                    t_sptoks = nlp(asp_term.get('term').decode())
                    if len(t_sptoks) > max_aspect_len:
                        max_aspect_len = len(t_sptoks)
        word_count = Counter(words).most_common()
	#print('word_count se la: %s' % word_count)
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)
		#print('word2id se la: %s' % word2id)
        with open(save_fname, 'w') as f:
            f.write('length %s %s\n' % (max_sentence_len, max_aspect_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))
    ###########
    #print('max_sentence_len se la: %s' % max_sentence_len)            
    print('There are %s words in the dataset, the max length of sentence is %s, and the max length of aspect is %s' % (len(word2id), max_sentence_len, max_aspect_len))
	
    return word2id, max_sentence_len, max_aspect_len

def get_loc_info(sptoks, from_id, to_id, sentiment_data, t_sptoks):
    aspect = []
    
	#tinh vi tri cua t_sptoks
#    for sptok in sptoks:
#	#print('sptok se la: %s' % sptok) 
#	#print('sptok.idx se la: %s' % sptok.idx)  #sptok.idx chinh la vi tri bat dau cua tu 
#	#print('sptok.idx se la: %s' % sptok.idx) 
#	#print('len(sptok.text)se la: %s' % len(sptok.text)) 
#       if sptok.idx < to_id and sptok.idx + len(sptok.text) > from_id:
#	    #print('sptok se la: %s' % sptok) 
#	    #print('sptok.idx se la: %s' % sptok.idx) #chinh la vi tri bat dau cua tu 
#	    #print('ssptok.i se la: %s' % sptok.i) 	#vi tri xuat hien cua term
#	    #print('from_id se la: %s' % from_id)
#           aspect.append(sptok.i)
#    indexaps = aspect[0]
    indexaps = 0
    count = 0
    for sptok in sptoks:
	if str(sptok) == str(t_sptoks):
	    indexaps=count
	    break
	count+=1
    count = 0
    loc_info = []
    #print('aspect se la: %s' % indexaps)
    #print('len(sptoks) se la: %s' % len(sptoks))
    #print(list(enumerate(sptoks))) #danh so thu tu cho tung tu trong cau
    #for i in aspect:
    #    print(i)
    
    for _i, sptok in enumerate(sptoks):
	#print('sptok se la: %s' % sptok) 
	#print('len(sptoks) se la: %s' % len(sptoks))
	#print('_i se la: %s' % _i) 
	#print('indexaps se la: %s' % indexaps) 

	weight = 1 - (abs(_i - indexaps)*1.0 / len(sptoks))
	if _i == indexaps:
	    weight = 0
        loc_info.append(weight) 

    maxW = max([i for i in loc_info])
    #print('sptoks se la: %s' % sptoks) 

    #doc file luu lai thanh danh sach cho de
    arrsentiword = []
    with open(sentiment_data, 'rb') as f:
       	for line in f:
	    line = line.decode('utf-8')
            content = line.strip()
    	    arrsentiword.append(content)

    countasp = 1
    for _i, sptok in enumerate(sptoks):
	for line in arrsentiword:
	    if content == str(sptok):
	        weight = maxW + (0.02*1.0/countasp)
	        loc_info[_i] = weight
	        countasp+=1
	        break

    #Cap nhat lai tinh danh gia
#    countasp = 1
#    for _i, sptok in enumerate(sptoks):
#	
#	with open(sentiment_data, 'rb') as f:
#       	    for line in f:
#            	line = line.decode('utf-8')
#            	content = line.strip()
		#print('content la %s' % content)
		#print('sptok la %s' % sptok)
#            	if content == str(sptok):
#		    weight = maxW + (0.05*1.0/countasp)
#		    loc_info[_i] = weight
#		    countasp+=1
#		    break

    return loc_info

def read_data(fname, word2id, max_sentence_len, max_aspect_len, save_fname, pre_processed, sentiment_data):
    sentences, aspects, sentence_lens, sentence_locs, labels = [], [], [], [], []
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            sentences.append(ast.literal_eval(lines[i]))
            aspects.append(ast.literal_eval(lines[i + 1]))
            sentence_lens.append(ast.literal_eval(lines[i + 2]))
            sentence_locs.append(ast.literal_eval(lines[i + 3]))
            labels.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)
	
	print('parse train data thanh cay roi lay root ...')
        tree = ET.parse(fname)
        root = tree.getroot()
        with open(save_fname, 'w') as f:
	    count=0
            for sentence in root:
                sptoks = nlp(sentence.find('text').text.decode())
		#print('sptoks se la: %s' % sptoks)
                if len(sptoks.text.strip()) != 0:
		    #print('Khoi tao ids ...')
                    ids = []
                    for sptok in sptoks:
			#print('sptok se la: %s' % sptok)
                        if sptok.text.lower() in word2id:
                            ids.append(word2id[sptok.text.lower()])
                    for asp_terms in sentence.iter('aspectTerms'):
                        for asp_term in asp_terms.findall('aspectTerm'):
                            if asp_term.get('polarity') == 'conflict':
                                continue
                            t_sptoks = nlp(asp_term.get('term').decode())
			    #print('t_sptoks se la: %s' % t_sptoks)
			    f.write("%s\n" % t_sptoks)
                            t_ids = []
                            for t_sptok in t_sptoks:
                                if t_sptok.text.lower() in word2id:
                                    t_ids.append(word2id[t_sptok.text.lower()])
				
			    #print('t_sptoks se la: %s' % t_sptoks)
			    #print('[0] se la: %s' % [0])
                            #print('max_sentence_len se la: %s' % max_sentence_len)
 			    #print('len(ids) se la: %s' % len(ids))

			    #############Them luu cau de de biet
				
			    f.write("%s\n" % sptoks)
			    f.write("%s\n" % count)
			    count+=1

                            sentences.append(ids + [0] * (max_sentence_len - len(ids)))
			    #print('sentences se la: %s' % sentences)
                            f.write("%s\n" % sentences[-1])

                            aspects.append(t_ids + [0] * (max_aspect_len - len(t_ids)))
                            f.write("%s\n" % aspects[-1])

                            sentence_lens.append(len(sptoks))
                            f.write("%s\n" % sentence_lens[-1])

			    #print('Tim loc-info ...')			    
                            loc_info = get_loc_info(sptoks, int(asp_term.get('from')), int(asp_term.get('to')), sentiment_data, t_sptoks)
			    #print('loc_info se la: %s' % loc_info)
			    

                            #sentence_locs.append(loc_info + [1] * (max_sentence_len - len(loc_info)))
			    sentence_locs.append(loc_info + [0] * (max_sentence_len - len(loc_info)))
			    #print('sentence_locs se la: %s' % sentence_locs) #mot mang mang gia tri 0 1, 0 la tu co trong cau, 1 la ko co
			    #f.write("%s\n" % "aaa")
			    

                            f.write("%s\n" % sentence_locs[-1])
			    #sys.exit()

                            polarity = asp_term.get('polarity')
                            if polarity == 'negative':
                                labels.append([1, 0, 0])
                            elif polarity == 'neutral':
                                labels.append([0, 1, 0])
                            elif polarity == "positive":
                                labels.append([0, 0, 1])
                            f.write("%s\n" % labels[-1])

    print("Read %s sentences from %s" % (len(sentences), fname))
    #print('np.asarray(labels)se la: %s' % np.asarray(labels)) 	#tao thanh mang
    #print('len(labels)se la: %s' % len(labels)) 	#tao thanh mang
    #print('len(sentences)se la: %s' % len(sentences)) 	#tao thanh mang
    return np.asarray(sentences), np.asarray(aspects), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels)

def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.normal(0, 0.05, [len(word2id), embedding_dim])
    #print('word2vec se la: %s' % word2vec)

    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            #line = line.decode('utf-8')
            content = line.strip().split()
	    #print('content %s' % content[0])
	    #print('content %s' % content[1:])
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

def get_batch_index(length, batch_size, is_shuffle=True):
    #index = list(range(length-1))
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]

def get_batch_index1(length, batch_size, is_shuffle=True):
    index = list(range(length))
#    print("index bbb %s " % index)
#    yield index
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
#        #yield index[i * batch_size:(i + 1) * batch_size]
	yield index


