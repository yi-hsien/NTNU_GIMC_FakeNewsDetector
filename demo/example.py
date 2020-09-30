import sys
demo = sys.argv[0]
#print('<br>')
#print(demo)

import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
import pandas as pd
import json
import csv
import string
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils

#sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

print('<br>')
#print (sys.getdefaultencoding())    # utf-8
#print (sys.stdout.encoding)         # ANSI_X3.4-1968
#sess = tf.Session()
#params = len(sys.argv)
title_src1 = sys.argv[1]
#title_src1 = '駐港官員拒簽一中 徐國勇：台灣不屬於中國'
content_src1 = sys.argv[2]
#content_src1 = '（中央社記者王承中台北18日電）媒體報導指出，陸委會駐港官員因拒簽「一中」切結書被迫回台。內政部長徐國勇今天表示，沒有所謂台灣是中國一部分的「一中」原則，台灣是主權獨立的國家，不是中國的一部分，政府對於主權絕不讓步。根據路透社報導，陸委會駐香港辦事處官員獲港府告知，如果不簽署「一中」切結書，他們的簽證將無法延簽。陸委會駐港辦事處代理處長>高銘村因拒簽以致簽證未獲延期，已回到台灣。徐國勇今天晚間前往士林夜市使用自己的三倍券促進消費、拚經濟。對於駐港人員被迫回台，徐國勇在受訪時表示，相信陸委會會做很好的處理，台灣是>主權獨立的國家，不是中國的一部分；政府對於主權絕不讓步，沒有所謂台灣是中國的一部分的「一中」原則，他也不接受。徐國勇說：「我們是中華民國台灣，跟中華人民共和國沒有關係。」振興三>倍券上路邁入第4天，徐國勇今晚前往士林夜市使用三倍券。徐國勇表示，他小時候就從萬華到這邊，家裡在士林市場內賣滷肉飯，這裡是他長大的地方，很多攤販都是他的鄰居，士林夜市是他非常熟悉且感恩的地方。至於三倍券跟現金相比較，徐國勇認為，三倍券優惠比較多，到很多地方消費都會加碼、物超所值，但他更希望大家不要只把三倍券花完，還要拿出現金搭配使用，要多多消費、振興經>濟，也希望大家有時間能到士林夜市消費。隨後，在夜市商圈人員的引導下，徐國勇開始採買小吃，沿路遇到不少熟人並熱情地打招呼；在使用三倍券時，徐國勇不忘向攤商宣導三倍券不可以找零，並>強調三倍券的用途很廣，還可以當香油錢。（編輯：鄭雪文）1090718'

#title_src2 = title_src1.encode('utf-8').decode('utf-8')
#content_src2 = content_src1.encode("utf-8").decode("utf-8")
#print('tle=' ,title_src1)
#print('<br>')
#print(content_src1)

ws = WS("/home/csliao/tf01/code/data")
pos = POS("/home/csliao/tf01/code/data")
ner = NER("/home/csliao/tf01/code/data")

# Feature Extraction
result = []
    
# 1. procress tbl_nm
#if rawdata[0] == 'mygopen':
#    checked = 1
#else:
#    checked = 0

# parse title text
#title_sentence_list = list(title_src2) 
title_sentence_list = [title_src1,]
#title_sentence_list = list("thisisabook")
#title_sentence_list = [rawdata[4],]
#print(title_sentence_list[0].encode('utf-8', errors='surrogateescape').decode('utf-8'))

word_title_sentence_list = ws(title_sentence_list,
                              sentence_segmentation=True,
                              segment_delimiter_set = {',', '.', '/', '?', ';', ':', '[', ']', '<', '>', '{', '}',
                                                       '(', ')', '\\', '\'', '\"', '#', '!', '`', '~', '@', '$',
                                                       '%', '^', '&', '*', '-', '_', '，', '。', '？', '：', '；',
                                                       '、', '「', '」', '＼', '｜', '『', '』', '～', '！', '＠',
                                                       '＃', '＄', '％', '︿', '＆', '＊', '（', '）', '—'})

#print(type(word_title_sentence_list))
word_title_sentence_list2 = str(word_title_sentence_list).strip('[]')
word_title_sentence_list3 = str(word_title_sentence_list2.replace("'", ""))
#print("aaa=", len(word_title_sentence_list3))   # print each single word from the title
pos_title_sentence_list = pos(word_title_sentence_list)
entity_title_sentence_list = ner(word_title_sentence_list, pos_title_sentence_list)
#print('ddd=', entity_title_sentence_list)

# remove punctuation
title_wordlist1 = str(word_title_sentence_list3.replace(",", "").replace("¡C", "").replace("¡]", "").replace("¡^", ""))
title_wordlist2 = str(title_wordlist1.replace("¡A", "").replace("¡B", "").replace("(", "").replace(")", ""))
title_wordlist3 = str(title_wordlist2.replace("¡u", "").replace("¡v", "").replace("¡I", "").replace("¡G", ""))
title_wordlist = str(title_wordlist3).split()
#print ('ooo=', title_wordlist)

title_wordfreq = []
for title_w in title_wordlist:
    title_wordfreq.append(title_wordlist.count(title_w))

#print('bbbb=', title_wordfreq[0])
title_a = zip(title_wordlist, title_wordfreq)
#print('cccc=', title_a)

def print_word_pos_sentence_title(word_title_sentence, pos_title_sentence):
    assert len(word_title_sentence) == len(pos_title_sentence)
    for word_title, pos_title in zip(word_title_sentence, pos_title_sentence):
        print(f"{word_title}({pos_title})", end="\u3000")
    print("\n")
    #print()
    return

for title_i, title_sentence in enumerate(title_sentence_list):
    who_title_str = date_title_str = time_title_str = ''
    title_person = title_date = title_time = 0
    #print('aaaaa0922=', len(entity_title_sentence_list[0]))
    sorted_aa = sorted(entity_title_sentence_list[0], key=lambda x:x[1])
    s = set(tuple(l) for l in sorted_aa)
    sorted_bb = [list(t) for t in s]
    #print('qaz=', len(sorted_bb))
    #print('ddd=', sorted_bb[0][2], sorted_bb[1][2])
    for title_entity in range(len(sorted_bb)):
        #print('bbbbb0922=',title_entity)
        if sorted_bb[title_entity][2] == 'PERSON':
            if who_title_str != "":
                if sorted_bb[title_entity][3] not in who_title_str :
                    who_title_str = who_title_str + '/' + sorted_bb[title_entity][3]
                    title_person = title_person+1
            else:
                who_title_str = sorted_bb[title_entity][3]
                title_person = 1
            #print('Title_Who = ', who_title_str)
        if sorted_bb[title_entity][2] == 'DATE':
            if date_title_str != "":
                if sorted_bb[title_entity][3] not in date_title_str :
                    date_title_str = date_title_str + '/' + sorted_bb[title_entity][3]
                    title_date = title_date+1
            else:
                date_title_str = sorted_bb[title_entity][3]
                title_date = 1
            #print('Title_When = ', date_title_str)
        if sorted_bb[title_entity][2] == 'TIME':
            if time_title_str != "":
                if sorted_bb[title_entity][3] not in time_title_str :
                    time_title_str = time_title_str + '/' + sorted_bb[title_entity][3]
                    title_time = title_time+1
            else:
                time_title_str = sorted_bb[title_entity][3]
                title_time = 1
            #print('Title_When1 = ', time_title_str)
       
# process title part-of-speech count
# 53 tags for title
pos_t_A = pos_t_Caa = pos_t_Cab = pos_t_Cba = pos_t_Cbb = pos_t_D = pos_t_Da = pos_t_Dfa = pos_t_Dfb = 0
pos_t_Di = pos_t_Dk = pos_t_DM = pos_t_I = pos_t_Na = pos_t_Nb = pos_t_Nc = pos_t_Ncd = pos_t_Nd = pos_t_Nep = 0
pos_t_Neqa = pos_t_Neqb = pos_t_Nes = pos_t_Neu = pos_t_Nf = pos_t_Ng = pos_t_Nh = pos_t_Nv = pos_t_P = 0
pos_t_T = pos_t_VA = pos_t_VAC = pos_t_VB = pos_t_VC = pos_t_VCL = pos_t_VD = pos_t_VF = pos_t_VE = pos_t_VG = 0
pos_t_VH = pos_t_VHC = pos_t_VI = pos_t_VJ = pos_t_VK = pos_t_VL = pos_t_V_2 = pos_t_DE = pos_t_SHI = pos_t_FW = 0
pos_t_COLONCATEGORY = pos_t_COMMACATEGORY = pos_t_DASHCATEGORY = pos_t_DOTCATEGORY = pos_t_ETCCATEGORY = 0
pos_t_EXCLAMATIONCATEGORY = pos_t_PARENTHESISCATEGORY = pos_t_PAUSECATEGORY = pos_t_PERIODCATEGORY = 0
pos_t_QUESTIONCATEGORY = pos_t_SEMICOLONCATEGORY = pos_t_SPCHANGECATEGORY = pos_t_WHITESPACE = 0
title_pos_count = len(str(pos_title_sentence_list).split())
#print('word_title_sentence_list =', word_title_sentence_list)
#print('pos_title_sentence_list =', pos_title_sentence_list)
#print('title_pos_count =', title_pos_count)
#print('0=',type(pos_title_sentence_list))
#print('1=',pos_title_sentence_list[10])
pos_title_i = 0
for pos_title_i in range(int(title_pos_count)-1):
    if pos_title_sentence_list[0][pos_title_i] == 'A':
        pos_t_A = pos_t_A + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Caa':
        pos_t_Caa = pos_t_Caa + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Cab':
        pos_t_Cab = pos_t_Cab + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Cba':
        pos_t_Cba = pos_t_Cba + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Cbb':
        pos_t_Cbb = pos_t_Cbb + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'D':
        pos_t_D = pos_t_D + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Da':
        pos_t_Da = pos_t_Da + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Dfa':
        pos_t_Dfa = pos_t_Dfa + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Dfb':
        pos_t_Dfb = pos_t_Dfb + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Di':
        pos_t_Di = pos_t_Di + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Dk':
        pos_t_Dk = pos_t_Dk + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'DM':
        pos_t_DM = pos_t_DM + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'I':
        pos_t_I = pos_t_I + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Na':
        pos_t_Na = pos_t_Na + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nb':
        pos_t_Nb = pos_t_Nb + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nc':
        pos_t_Nc = pos_t_Nc + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Ncd':
        pos_t_Ncd = pos_t_Ncd + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nd':
        pos_t_Nd = pos_t_Nd + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nep':
        pos_t_Nep = pos_t_Nep + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Neqa':
        pos_t_Neqa = pos_t_Neqa + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Neqb':
        pos_t_Neqb = pos_t_Neqb + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nes':
        pos_t_Nes = pos_t_Nes + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Neu':
        pos_t_Neu = pos_t_Neu + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nf':
        pos_t_Nf = pos_t_Nf + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Ng':
        pos_t_Ng = pos_t_Ng + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nh':
        pos_t_Nh = pos_t_Nh + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'Nv':
        pos_t_Nv = pos_t_Nv + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'P':
        pos_t_P = pos_t_P + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'T':
        pos_t_T = pos_t_T + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VA':
        pos_t_VA = pos_t_VA + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VAC':
        pos_t_VAC = pos_t_VAC + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VB':
        pos_t_VB = pos_t_VB + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VC':
        pos_t_VC = pos_t_VC + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VCL':
        pos_t_VCL = pos_t_VCL + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VD':
        pos_t_VD = pos_t_VD + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VF':
        pos_t_VF = pos_t_VF + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VE':
        pos_t_VE = pos_t_VE + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VG':
        pos_t_VG = pos_t_VG + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VH':
        pos_t_VH = pos_t_VH + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VHC':
        pos_t_VHC = pos_t_VHC + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VI':
        pos_t_VI = pos_t_VI + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VJ':
        pos_t_VJ = pos_t_VJ + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VK':
        pos_t_VK = pos_t_VK + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'VL':
        pos_t_VL = pos_t_VL + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'V_2':
        pos_t_V_2 = pos_t_V_2 + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'DE':
        pos_t_DE = pos_t_DE + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'SHI':
        pos_t_SHI = pos_t_SHI + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'FW':
        pos_t_FW = pos_t_FW + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'COLONCATEGORY':
        pos_t_COLONCATEGORY = pos_t_COLONCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'COMMACATEGORY':
        pos_t_COMMACATEGORY = pos_t_COMMACATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'DASHCATEGORY':
        pos_t_DASHCATEGORY = pos_t_DASHCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'DOTCATEGORY':
        pos_t_DOTCATEGORY = pos_t_DOTCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'ETCCATEGORY':
        pos_t_ETCCATEGORY = pos_t_ETCCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'EXCLAMATIONCATEGORY':
        pos_t_EXCLAMATIONCATEGORY = pos_t_EXCLAMATIONCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'PARENTHESISCATEGORY':
        pos_t_PARENTHESISCATEGORY = pos_t_PARENTHESISCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'PAUSECATEGORY':
        pos_t_PAUSECATEGORY = pos_t_PAUSECATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'PERIODCATEGORY':
        pos_t_PERIODCATEGORY = pos_t_PERIODCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'QUESTIONCATEGORY':
        pos_t_QUESTIONCATEGORY = pos_t_QUESTIONCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'SEMICOLONCATEGORY':
        pos_t_SEMICOLONCATEGORY = pos_t_SEMICOLONCATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'SPCHANGECATEGORY':
        pos_t_SPCHANGECATEGORY = pos_t_SPCHANGECATEGORY + 1
    elif pos_title_sentence_list[0][pos_title_i] == 'WHITESPACE':
        pos_t_WHITESPACE = pos_t_WHITESPACE + 1

#print('pos_t_Na =', pos_t_Na)
#print('pos_t_Neqa = ', pos_t_Neqa)

# 4. procress content
# print(rawdata[5])   # print content

# parse content text
#content_sentence_list = list(content_src2)
content_sentence_list = [content_src1,]
#print('<br>')
#print('content :', content_src1)
word_content_sentence_list = ws(content_sentence_list,
                                sentence_segmentation=True,
                                segment_delimiter_set = {',', '.', '/', '?', ';', ':', '[', ']', '<', '>', '{', '}',
                                                       '(', ')', '\\', '\'', '\"', '#', '!', '`', '~', '@', '$',
                                                       '%', '^', '&', '*', '-', '_', '，', '。', '？', '：', '；',
                                                       '、', '「', '」', '＼', '｜', '『', '』', '～', '！', '＠',
                                                       '＃', '＄', '％', '︿', '＆', '＊', '（', '）', '—'})

#print(type(word_content_sentence_list))
word_content_sentence_list2 =  str(word_content_sentence_list).strip('[]')
word_content_sentence_list3 = str(word_content_sentence_list2.replace("'", ""))
#print("bbb=", word_content_sentence_list3)   # print each single word from the content
pos_content_sentence_list = pos(word_content_sentence_list)
entity_content_sentence_list = ner(word_content_sentence_list, pos_content_sentence_list)

# remove punctuation
content_wordlist1 = str(word_content_sentence_list3.replace(",", "").replace("¡C", "").replace("¡]", "").replace("¡^", ""))
content_wordlist2 = str(content_wordlist1.replace("¡A", "").replace("¡B", "").replace("(", "").replace(")", ""))
content_wordlist3 = str(content_wordlist2.replace("¡u", "").replace("¡v", "").replace("¡I", "").replace("¡G", ""))
content_wordlist = str(content_wordlist3).split()
#print ('ccc=', content_wordlist)

content_wordfreq = []
for content_w in content_wordlist:
    content_wordfreq.append(content_wordlist.count(content_w))

content_a = zip(content_wordlist, content_wordfreq)

def print_word_pos_sentence_content(word_content_sentence, pos_content_sentence):
    assert len(word_content_sentence) == len(pos_content_sentence)
    for word_content, pos_content in zip(word_content_sentence, pos_content_sentence):
        print(f"{word_content}({pos_content})", end="\u3000")
    print("\n")
    #print()
    return

for content_i, content_sentence in enumerate(content_sentence_list):
    #print_word_pos_sentence_content(word_content_sentence_list[content_i],  pos_content_sentence_list[content_i])
    who_content_str = date_content_str = time_content_str = ''
    content_person = content_date = content_time = 0
    for content_entity in sorted(entity_content_sentence_list[content_i]):
        #print(content_entity[2:])
        if content_entity[2] == 'PERSON':
            if who_content_str != "":
                if content_entity[3] not in who_content_str :
                    who_content_str = who_content_str + '/' + content_entity[3]
                    content_person = content_person+1
            else:
                who_content_str = content_entity[3]
                content_person = 1
            #print('Content_Who = ', content_person)
        if content_entity[2] == 'DATE':
            if date_content_str != "":
                if content_entity[3] not in date_content_str :
                    date_content_str = date_content_str + '/' + content_entity[3]
                    content_date = content_date+1
            else:
                date_content_str = content_entity[3]
                content_date = 1
            #print('Content_When = ', content_date)
        if content_entity[2] == 'TIME':
            if time_content_str != "":
                if content_entity[3] not in time_content_str :
                    time_content_str = time_content_str + '/' + content_entity[3]
                    content_time = content_time+1
            else:
                time_content_str = content_entity[3]
                content_time = 1
            #print('Content_When1 = ', content_time)

# process title_num in content_num / title_num
title_content = rel_title_content = 0
rel_tc = ''
for rel_tc in title_wordlist:
    if rel_tc in content_wordlist:
        title_content = title_content+1
title_count = len(title_wordlist)
rel_title_content = title_content/title_count
#print('rel_title_content =', rel_title_content)

# process content part-of-speech count
# 53 tags for content
pos_c_A = pos_c_Caa = pos_c_Cab = pos_c_Cba = pos_c_Cbb = pos_c_D = pos_c_Da = pos_c_Dfa = pos_c_Dfb = 0
pos_c_Di = pos_c_Dk = pos_c_DM = pos_c_I = pos_c_Na = pos_c_Nb = pos_c_Nc = pos_c_Ncd = pos_c_Nd = pos_c_Nep = 0
pos_c_Neqa = pos_c_Neqb = pos_c_Nes = pos_c_Neu = pos_c_Nf = pos_c_Ng = pos_c_Nh = pos_c_Nv = pos_c_P = 0
pos_c_T = pos_c_VA = pos_c_VAC = pos_c_VB = pos_c_VC = pos_c_VCL = pos_c_VD = pos_c_VF = pos_c_VE = pos_c_VG = 0
pos_c_VH = pos_c_VHC = pos_c_VI = pos_c_VJ = pos_c_VK = pos_c_VL = pos_c_V_2 = pos_c_DE = pos_c_SHI = pos_c_FW = 0
pos_c_COLONCATEGORY = pos_c_COMMACATEGORY = pos_c_DASHCATEGORY = pos_c_DOTCATEGORY = pos_c_ETCCATEGORY = 0
pos_c_EXCLAMATIONCATEGORY = pos_c_PARENTHESISCATEGORY = pos_c_PAUSECATEGORY = pos_c_PERIODCATEGORY = 0
pos_c_QUESTIONCATEGORY = pos_c_SEMICOLONCATEGORY = pos_c_SPCHANGECATEGORY = pos_c_WHITESPACE = 0

#title_pos_count = len(str(word_title_sentence_list).split())
content_pos_count = len(str(pos_content_sentence_list).split())
#print('word_content_sentence_list =', word_content_sentence_list)
#print('pos_content_sentence_list =', pos_content_sentence_list)
#print('content_pos_count =', content_pos_count)
#print('2=',pos_content_sentence_list[0])
pos_content_i = 0
for pos_content_i in range(int(content_pos_count)-1):
    if pos_content_sentence_list[0][pos_content_i] == 'A':
        pos_c_A = pos_c_A + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Caa':
        pos_c_Caa = pos_c_Caa + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Cab':
        pos_c_Cab = pos_c_Cab + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Cba':
        pos_c_Cba = pos_c_Cba + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Cbb':
        pos_c_Cbb = pos_c_Cbb + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'D':
        pos_c_D = pos_c_D + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Da':
        pos_c_Da = pos_c_Da + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Dfa':
        pos_c_Dfa = pos_c_Dfa + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Dfb':
        pos_c_Dfb = pos_c_Dfb + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Di':
        pos_c_Di = pos_c_Di + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Dk':
        pos_c_Dk = pos_c_Dk + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'DM':
        pos_c_DM = pos_c_DM + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'I':
        pos_c_I = pos_c_I + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Na':
        pos_c_Na = pos_c_Na + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nb':
        pos_c_Nb = pos_c_Nb + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nc':
        pos_c_Nc = pos_c_Nc + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Ncd':
        pos_c_Ncd = pos_c_Ncd + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nd':
        pos_c_Nd = pos_c_Nd + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nep':
        pos_c_Nep = pos_c_Nep + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Neqa':
        pos_c_Neqa = pos_c_Neqa + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Neqb':
        pos_c_Neqb = pos_c_Neqb + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nes':
        pos_c_Nes = pos_c_Nes + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Neu':
        pos_c_Neu = pos_c_Neu + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nf':
        pos_c_Nf = pos_c_Nf + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Ng':
        pos_c_Ng = pos_c_Ng + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nh':
        pos_c_Nh = pos_c_Nh + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'Nv':
        pos_c_Nv = pos_c_Nv + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'P':
        pos_c_P = pos_c_P + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'T':
        pos_c_T = pos_c_T + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VA':
        pos_c_VA = pos_c_VA + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VAC':
        pos_c_VAC = pos_c_VAC + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VB':
        pos_c_VB = pos_c_VB + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VC':
        pos_c_VC = pos_c_VC + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VCL':
        pos_c_VCL = pos_c_VCL + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VD':
        pos_c_VD = pos_c_VD + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VF':
        pos_c_VF = pos_c_VF + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VE':
        pos_c_VE = pos_c_VE + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VG':
        pos_c_VG = pos_c_VG + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VH':
        pos_c_VH = pos_c_VH + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VHC':
        pos_c_VHC = pos_c_VHC + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VI':
        pos_c_VI = pos_c_VI + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VJ':
        pos_c_VJ = pos_c_VJ + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VK':
        pos_c_VK = pos_c_VK + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'VL':
        pos_c_VL = pos_c_VL + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'V_2':
        pos_c_V_2 = pos_c_V_2 + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'DE':
        pos_c_DE = pos_c_DE + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'SHI':
        pos_c_SHI = pos_c_SHI + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'FW':
        pos_c_FW = pos_c_FW + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'COLONCATEGORY':
        pos_c_COLONCATEGORY = pos_c_COLONCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'COMMACATEGORY':
        pos_c_COMMACATEGORY = pos_c_COMMACATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'DASHCATEGORY':
        pos_c_DASHCATEGORY = pos_c_DASHCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'DOTCATEGORY':
        pos_c_DOTCATEGORY = pos_c_CATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'ETCCATEGORY':
        pos_c_ETCCATEGORY = pos_c_CATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'EXCLAMATIONCATEGORY':
        pos_c_EXCLAMATIONCATEGORY = pos_c_EXCLAMATIONCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'PARENTHESISCATEGORY':
        pos_c_PARENTHESISCATEGORY = pos_c_PARENTHESISCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'PAUSECATEGORY':
        pos_c_PAUSECATEGORY = pos_c_PAUSECATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'PERIODCATEGORY':
        pos_c_PERIODCATEGORY = pos_c_PERIODCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'QUESTIONCATEGORY':
        pos_c_QUESTIONCATEGORY = pos_c_QUESTIONCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'SEMICOLONCATEGORY':
        pos_c_SEMICOLONCATEGORY = pos_c_SEMICOLONCATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'SPCHANGECATEGORY':
        pos_c_SPCHANGECATEGORY = pos_c_SPCHANGECATEGORY + 1
    elif pos_content_sentence_list[0][pos_content_i] == 'WHITESPACE':
        pos_c_WHITESPACE = pos_c_WHITESPACE + 1


label = 1 
content_fin = title_src1, title_person, title_date, title_time, content_person, content_date, content_time, pos_t_A, pos_t_Caa, pos_t_Cab, pos_t_Cba, pos_t_Cbb, pos_t_D, pos_t_Da, pos_t_Dfa, pos_t_Dfb, pos_t_Di, pos_t_Dk, pos_t_DM, pos_t_I, pos_t_Na, pos_t_Nb, pos_t_Nc, pos_t_Ncd, pos_t_Nd, pos_t_Nep, pos_t_Neqa, pos_t_Neqb, pos_t_Nes, pos_t_Neu, pos_t_Nf, pos_t_Ng, pos_t_Nh, pos_t_Nv, pos_t_P, pos_t_T, pos_t_VA, pos_t_VAC, pos_t_VB, pos_t_VC, pos_t_VCL, pos_t_VD, pos_t_VF, pos_t_VE, pos_t_VG, pos_t_VH, pos_t_VHC, pos_t_VI, pos_t_VJ, pos_t_VK, pos_t_VL, pos_t_V_2, pos_t_DE, pos_t_SHI, pos_t_FW, pos_t_COLONCATEGORY, pos_t_COMMACATEGORY, pos_t_DASHCATEGORY, pos_t_DOTCATEGORY, pos_t_ETCCATEGORY, pos_t_EXCLAMATIONCATEGORY, pos_t_PARENTHESISCATEGORY, pos_t_PAUSECATEGORY, pos_t_PERIODCATEGORY, pos_t_QUESTIONCATEGORY, pos_t_SEMICOLONCATEGORY, pos_t_SPCHANGECATEGORY, pos_t_WHITESPACE, pos_c_A, pos_c_Caa, pos_c_Cab, pos_c_Cba, pos_c_Cbb, pos_c_D, pos_c_Da, pos_c_Dfa, pos_c_Dfb, pos_c_Di, pos_c_Dk, pos_c_DM, pos_c_I, pos_c_Na, pos_c_Nb, pos_c_Nc, pos_c_Ncd, pos_c_Nd, pos_c_Nep, pos_c_Neqa, pos_c_Neqb, pos_c_Nes, pos_c_Neu, pos_c_Nf, pos_c_Ng, pos_c_Nh, pos_c_Nv, pos_c_P, pos_c_T, pos_c_VA, pos_c_VAC, pos_c_VB, pos_c_VC, pos_c_VCL, pos_c_VD, pos_c_VF, pos_c_VE, pos_c_VG, pos_c_VH, pos_c_VHC, pos_c_VI, pos_c_VJ, pos_c_VK, pos_c_VL, pos_c_V_2, pos_c_DE, pos_c_SHI, pos_c_FW, pos_c_COLONCATEGORY, pos_c_COMMACATEGORY, pos_c_DASHCATEGORY, pos_c_DOTCATEGORY, pos_c_ETCCATEGORY, pos_c_EXCLAMATIONCATEGORY, pos_c_PARENTHESISCATEGORY, pos_c_PAUSECATEGORY, pos_c_PERIODCATEGORY, pos_c_QUESTIONCATEGORY, pos_c_SEMICOLONCATEGORY, pos_c_SPCHANGECATEGORY, pos_c_WHITESPACE, rel_title_content, label 
#content_fin = rawdata[1], rawdata[2], rawdata[3], title_person, title_date, title_time, content_person, content_date, content_time, rel_title_content,

result.append(content_fin)
columns = ['title', 'title_person', 'title_date', 'title_time', 'content_person', 'content_date', 'content_time', 'pos_t_A', 'pos_t_Caa', 'pos_t_Cab', 'pos_t_Cba', 'pos_t_Cbb', 'pos_t_D', 'pos_t_Da', 'pos_t_Dfa', 'pos_t_Dfb', 'pos_t_Di', 'pos_t_Dk', 'pos_t_DM', 'pos_t_I', 'pos_t_Na', 'pos_t_Nb', 'pos_t_Nc', 'pos_t_Ncd', 'pos_t_Nd', 'pos_t_Nep', 'pos_t_Neqa', 'pos_t_Neqb', 'pos_t_Nes', 'pos_t_Neu', 'pos_t_Nf', 'pos_t_Ng', 'pos_t_Nh', 'pos_t_Nv', 'pos_t_P', 'pos_t_T', 'pos_t_VA', 'pos_t_VAC', 'pos_t_VB', 'pos_t_VC', 'pos_t_VCL', 'pos_t_VD', 'pos_t_VF', 'pos_t_VE', 'pos_t_VG', 'pos_t_VH', 'pos_t_VHC', 'pos_t_VI', 'pos_t_VJ', 'pos_t_VK', 'pos_t_VL', 'pos_t_V_2', 'pos_t_DE', 'pos_t_SHI', 'pos_t_FW', 'pos_t_COLONCATEGORY', 'pos_t_COMMACATEGORY', 'pos_t_DASHCATEGORY', 'pos_t_DOTCATEGORY', 'pos_t_ETCCATEGORY', 'pos_t_EXCLAMATIONCATEGORY', 'pos_t_PARENTHESISCATEGORY', 'pos_t_PAUSECATEGORY', 'pos_t_PERIODCATEGORY', 'pos_t_QUESTIONCATEGORY', 'pos_t_SEMICOLONCATEGORY', 'pos_t_SPCHANGECATEGORY', 'pos_t_WHITESPACE', 'pos_c_A', 'pos_c_Caa', 'pos_c_Cab', 'pos_c_Cba', 'pos_c_Cbb', 'pos_c_D', 'pos_c_Da', 'pos_c_Dfa', 'pos_c_Dfb', 'pos_c_Di', 'pos_c_Dk', 'pos_c_DM', 'pos_c_I', 'pos_c_Na', 'pos_c_Nb', 'pos_c_Nc', 'pos_c_Ncd', 'pos_c_Nd', 'pos_c_Nep', 'pos_c_Neqa', 'pos_c_Neqb', 'pos_c_Nes', 'pos_c_Neu', 'pos_c_Nf', 'pos_c_Ng', 'pos_c_Nh', 'pos_c_Nv', 'pos_c_P', 'pos_c_T', 'pos_c_VA', 'pos_c_VAC', 'pos_c_VB', 'pos_c_VC', 'pos_c_VCL', 'pos_c_VD', 'pos_c_VF', 'pos_c_VE', 'pos_c_VG', 'pos_c_VH', 'pos_c_VHC', 'pos_c_VI', 'pos_c_VJ', 'pos_c_VK', 'pos_c_VL', 'pos_c_V_2', 'pos_c_DE', 'pos_c_SHI', 'pos_c_FW', 'pos_c_COLONCATEGORY', 'pos_c_COMMACATEGORY', 'pos_c_DASHCATEGORY', 'pos_c_DOTCATEGORY', 'pos_c_ETCCATEGORY', 'pos_c_EXCLAMATIONCATEGORY', 'pos_c_PARENTHESISCATEGORY', 'pos_c_PAUSECATEGORY', 'pos_c_PERIODCATEGORY', 'pos_c_QUESTIONCATEGORY', 'pos_c_SEMICOLONCATEGORY', 'pos_c_SPCHANGECATEGORY', 'pos_c_WHITESPACE', 'rel_title_content', 'label']
#columns = ['website', 'date', 'title', 'title_person', 'title_date', 'title_time', 'content_person', 'contente_date', 'content_time', 'rel_title_content',]

df_feature = pd.DataFrame(result, columns=columns)


#print('<br>')
#print(df_feature)
np_feature1 = df_feature.loc[:, ['title_person', 'title_date', 'title_time', 'content_person', 'content_date', 'content_time', 'pos_t_D', 'pos_t_Dfa', 'pos_t_Na', 'pos_t_Nb', 'pos_t_Nc', 'pos_t_Nd', 'pos_t_Nf', 'pos_t_P', 'pos_t_VA', 'pos_t_VC', 'pos_t_VE', 'pos_t_VH', 'pos_t_VJ', 'pos_t_VK', 'pos_t_DE', 'pos_t_FW', 'pos_t_COLONCATEGORY', 'pos_t_COMMACATEGORY', 'pos_t_EXCLAMATIONCATEGORY', 'pos_t_PARENTHESISCATEGORY', 'pos_t_QUESTIONCATEGORY', 'pos_t_WHITESPACE', 'pos_c_A', 'pos_c_Caa', 'pos_c_Cab', 'pos_c_Cbb', 'pos_c_D', 'pos_c_Da', 'pos_c_Dfa', 'pos_c_Di', 'pos_c_Dk', 'pos_c_Na', 'pos_c_Nb', 'pos_c_Nc', 'pos_c_Ncd', 'pos_c_Nd', 'pos_c_Nep', 'pos_c_Neqa', 'pos_c_Neqb', 'pos_c_Nes', 'pos_c_Neu', 'pos_c_Nf', 'pos_c_Ng', 'pos_c_Nh', 'pos_c_Nv', 'pos_c_P', 'pos_c_T', 'pos_c_VA', 'pos_c_VAC', 'pos_c_VB', 'pos_c_VC', 'pos_c_VCL', 'pos_c_VD', 'pos_c_VF', 'pos_c_VE', 'pos_c_VG', 'pos_c_VH', 'pos_c_VHC', 'pos_c_VI', 'pos_c_VJ', 'pos_c_VK', 'pos_c_VL', 'pos_c_V_2', 'pos_c_DE', 'pos_c_SHI', 'pos_c_FW', 'pos_c_COLONCATEGORY', 'pos_c_COMMACATEGORY', 'pos_c_DASHCATEGORY', 'pos_t_EXCLAMATIONCATEGORY', 'pos_c_PARENTHESISCATEGORY', 'pos_c_PAUSECATEGORY', 'pos_c_PERIODCATEGORY', 'pos_c_QUESTIONCATEGORY', 'pos_c_SEMICOLONCATEGORY', 'pos_c_WHITESPACE', 'rel_title_content']].values
print('<br>')
print('feature_value =')
print('<br>')
print(np_feature1[0])
print('<br>')
#print('aaa <br>')
np_label1 = df_feature.loc[:, ['label']].values
#print(np_label1[13])
#print('dddd<br>')

from keras.models import load_model
new_model = load_model('/home/csliao/tf01/code/ntnu_ai_model.h5')
#new_model = load_model('/home/csliao/tf01/code/ntnu_aimodel0611.h5')
#print('bbb <br>')
test_result = new_model.predict(
    np_feature1,
    batch_size=100,
    verbose=0,
    steps=None)
accuracy = test_result[0]
#print('新聞可信度 Fake News Credibility rate = ')
print('<br>')
print('result = ')
print(accuracy)
#json_str = json.dumps(str(accuracy))
#print(unicode('新聞可信度:', encoding='utf-8'), json_str)

