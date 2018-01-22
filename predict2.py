#create table tweets (username VARCHAR(15), date DATETIME, retweets INT, favorites INT, text VARCHAR(150), geo VARCHAR(50), mentions VARCHAR(50), hashtags VARCHAR(50), id VARCHAR(20), permalink VARCHAR(100));
#for i in 31 {1..9}; do mysql -e "load data local infile 'dt_$i.csv' into table tweets fields terminated by ';' ignore 1 lines " twitter -u root  ; done

import codecs, re
import numpy as np
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import classification_report
#from mpl_toolkits.basemap import Basemap
#from matplotlib.patches import Polygon
#import matplotlib.pyplot as plt
#import stateplane
from collections import Counter
from wordcloud import WordCloud
import sys
from sklearn.model_selection import cross_val_score
import itertools
from langdetect import detect

def split_seq(iterable, size):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))

        
def xvalidation(pos, neg, neu, n):
    c_val = 5.0
    npos = len(pos)
    nneu = len(neu)
    nneg = len(neg)
    
    newpos = list(split_seq(pos, int(npos / n)))
    newneu = list(split_seq(neu, int(nneu / n)))
    newneg = list(split_seq(neg, int(nneg / n)))

    acc = []
    for i in range(1, n):
        #print(npos, nneu, nneg, len(newpos), len(newneu), len(newneg))
        test_data = newpos[i] + newneg[i] + newneu[i]
        test_labels = ['pos'] * len(newpos[i]) + ['neg'] * len(newneg[i])\
                      + ['neu'] * len(newneu[i])
        for j in range(0, n):
            if j != i:
                if 'pos2' not in vars():
                    pos2 = newpos[j]
                    neu2 = newneu[j]
                    neg2 = newneg[j]
                else:
                    pos2 = pos2 + newpos[j]
                    neu2 = neu2 + newneu[j]
                    neg2 = neg2 + newneg[j]
        train_data = pos2 + neu2 + neg2
        train_labels = ['pos'] * len(pos2) + ['neg'] * len(neg2) + \
                       ['neu'] * len(neu2)
        del pos2, neu2, neg2

        vectorizer = TfidfVectorizer(min_df=0, max_df = 0.9, \
                                     sublinear_tf=True, use_idf=True)
        train_vectors = vectorizer.fit_transform(train_data)
        test_vectors = vectorizer.transform(test_data)
        classifier_liblinear = svm.LinearSVC(C = c_val)
        classifier_liblinear.fit(train_vectors, train_labels)
        prediction_liblinear = classifier_liblinear.predict(test_vectors)
        rprt = classification_report(test_labels, \
                                     prediction_liblinear).split('\n')
        t = map(str.split, rprt)
        acc.append(float(list(t)[2][1]))

    return acc

    
def train_test(pos, neg, neu):
    c_val = 1.0
    npos = len(pos)
    nneu = len(neu)
    nneg = len(neg)

    for f in range(1, 11, 1):
#        acc = 0.0
        for i in range(20):
            np.random.shuffle(pos)
            np.random.shuffle(neg)
            np.random.shuffle(neu)

            newpos = pos[:int(0.1 * f * npos)]
            newneu = neu[:int(0.1 * f * nneu)]
            newneg = neg[:int(0.1 * f * nneg)]
        
            poscut = int(0.9 * len(newpos))
            neucut = int(0.9 * len(newneu))
            negcut = int(0.9 * len(newneg))

            a = xvalidation(newpos, newneg, newneu, 9)
            if 'acc' not in vars():
                acc = a
            else:
                acc = acc + a
        print(f*10, np.mean(acc), np.std(acc))
        del acc

    return
            
#            train_data = newpos[:poscut] + newneg[:negcut] + newneu[:neucut]
#            test_data = newpos[poscut:] + newneg[negcut:] + newneu[neucut:]
#            train_labels = ['pos'] * poscut + ['neg'] * \
#                           negcut + ['neu'] * neucut
#            test_labels = ['pos'] * (len(newpos) - poscut) + \
#                          ['neg'] * (len(newneg) - negcut) + \
#                          ['neu'] * (len(newneu) - neucut)
        
#            vectorizer = TfidfVectorizer(min_df=0, max_df = 0.9, \
#                                         sublinear_tf=True, use_idf=True)
#            train_vectors = vectorizer.fit_transform(train_data)
#            test_vectors = vectorizer.transform(test_data)
        
            # Perform classification with SVM, kernel=rbf
            #classifier_rbf = svm.SVC(C = c_val) #LinearSVC()
            #classifier_rbf.fit(train_vectors, train_labels)
            #prediction_rbf = classifier_rbf.predict(test_vectors)
            #print('kernel=rbf')
            #print(classification_report(test_labels, prediction_rbf))
            
            # Perform classification with SVM, kernel=linear
            #classifier_linear = svm.SVC(kernel='linear', C = c_val)
            #classifier_linear.fit(train_vectors, train_labels)
            #prediction_linear = classifier_linear.predict(test_vectors)
            #print('kernel=linear')
            #print(classification_report(test_labels, prediction_linear))
            
            # Perform classification with SVM, kernel=linear
            #for c_val in [0.1, 1.0, 5.0, 10.0, 100.0]:
#            classifier_liblinear = svm.LinearSVC(C = c_val)
#            classifier_liblinear.fit(train_vectors, train_labels)
#            prediction_liblinear = classifier_liblinear.predict(test_vectors)
            #print('kernel=linear using LinearSVC with C = ', c_val)
            #print(classification_report(test_labels, prediction_liblinear))
#            rprt = classification_report(test_labels, \
#                                         prediction_liblinear).split('\n')
#            t = map(str.split, rprt)
#            acc = acc + float(list(t)[2][1])
            #scores = cross_val_score(classifier_liblinear, \
            #                         iris.data, iris.target, cv=5)
            #Report structure
            #Col's: 'precision', 'recall', 'f1-score', 'support'
            #Rows:  'neg', 'neu', 'pos'
            #for line in rprt:
            #    wrds = line.split()
            #    if len(wrds) > 1:
            #        if wrds[0] == 'neg':
            #            acc = acc + float(wrds[1])

#        print(f, i, acc / i)


def plot_wordcloud(wbag, capt):
    wordcloud = WordCloud(relative_scaling = 0.5, height = 400).\
                generate(' '.join(wbag))
    plt.figure(capt)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('wc_' + capt + '.jpg')

    
def filter_tw(t):
    txt = re.sub(r'@.*:', '', t, flags=re.MULTILINE)
    txt = re.sub(r'@\w+', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'@', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'#\w+', '', txt, flags=re.MULTILINE)
    txt = re.sub(r':', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'"', '', txt, flags=re.MULTILINE)
    txt = re.sub(r',', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'\'', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'\?', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'!', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'-', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'\/', '', txt, flags=re.MULTILINE)
    txt = re.sub(r';', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'\.', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'\.+', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(' +',' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'\&amp','', txt, flags=re.MULTILINE)
    txt = re.sub(r'\$','', txt, flags=re.MULTILINE)
    txt = re.sub(r'\.', '', txt)
    return txt


def get_state(lon, lat):
    st = stateplane.identify(lon, lat, fmt = 'short')
    #if '_' in st:
    #    s = st.split('_')[0]
    #else:
    #    s = st
    return st[0:2]   
    

def abv_to_state(abv):
    states = {'AK' : 'Alaska', 'AR' : 'Arkansas', 'AZ' : 'Arizona', \
              'CA' : 'California', 'CO' : 'Colorado', \
              'CT' : 'Connecticut', \
              'DC' : 'District of Columbia', 'DE' : 'Delaware', \
              'FL' : 'Florida', 'GA' : 'Georgia', 'HA' : 'Hawaii', \
              'IA' : 'Iowa', 'ID' : 'Idaho', 'IL' : 'Illinois', \
              'IN' : 'Indiana', 'KS' : 'Kansas', 'KY' : 'Kentucky', \
              'LA' : 'Louisiana', 'MA' : 'Massachusetts', \
              'MD' : 'Maryland', 'ME' : 'Maine', 'MI' : 'Michigan', \
              'MN' : 'Minnesota', 'MO' : 'Missouri', \
              'MS' : 'Mississippi', \
              'MT' : 'Montana', 'NC' : 'North Carolina', \
              'ND' : 'North Dakota', 'NE' : 'Nebraska', \
              'NH' : 'New Hampshire', 'NJ' : 'New Jersey', \
              'NM' : 'New Mexico', 'NV' : 'Nevada', 'NY' : 'New York', \
              'OH' : 'Ohio', 'OK' : 'Oklahoma', 'OR' : 'Oregon', \
              'PA' : 'Pennsylvania', \
              'PR' : 'Puerto Rico', 'RI' : 'Rhode Island', \
              'SC' : 'South Carolina', \
              'SD' : 'South Dakota', 'TN' : 'Tennessee', 'TX' : 'Texas', \
              'UT' : 'Utah', 'VA' : 'Virginia', 'VT' : 'Vermont', \
              'WA' : 'Washington', 'WI' : 'Wisconsin', \
              'WV' : 'West Virginia', 'WY' : 'Wyoming'}
    if abv in states.keys():
        return states[abv]
    else:
        return 'NAS'


def state_check(str):
    st_short = ['AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', \
                'HA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', \
                'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', \
                'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', \
                'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', \
                'WY', 'N.Y.', 'N.Y', 'D.C.', 'D.C']
    st_long = ['Alaska', 'Arkansas', 'Arizona', 'California', 'Colorado', \
               'Connecticut', 'Delaware', \
               'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', \
               'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', \
               'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', \
               'Mississippi', 'Montana', \
               'Nebraska', \
               'Nevada', 'Ohio', 'Oklahoma', 'Oregon','Pennsylvania', \
               'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', \
               'Wisconsin', 'Wyoming']

    #st_short = [st.lower() for st in st_short]
    #st_long = [st.lower() for st in st_long]
    #str = str.lower()
    str = re.sub(r'[^a-zA-Z.]', ' ', str)
    #str = str.title()
    wrds = str.split()
    for i, wrd in enumerate(wrds[:-1]):
        if wrd in ['New'] and wrds[i+1] in ['Hampshire', 'Jersey', 'Mexico', 'York']:
            add_state = wrd + ' ' + wrds[i+1]
            return add_state
        else:
            if wrd in ['North'] and wrds[i+1] in ['Carolina', 'Dakota']:
                add_state = wrd + ' ' + wrds[i+1]
                return add_state
            else:
                if wrd in ['South'] and wrds[i+1] in ['Carolina', 'Dakota']:
                    add_state = wrd + ' ' + wrds[i+1]
                    return add_state
                else:
                    if wrd in ['West'] and wrds[i+1] in ['Virginia']:
                        add_state = wrd + ' ' + wrds[i+1]
                        return add_state
                    else:
                        if wrd in ['Rhode'] and wrds[i+1] in ['Island']:
                            add_state = wrd + ' ' + wrds[i+1]
                            return add_state

    if ['District'] in wrds:
        idx = wrds.index('District')
        if idx < len(wrds) - 2 and \
           wrds[idx+1] == 'Of' and wrds[idx+2] == 'Columbia':
            add_state = 'District of Columbia'
            return add_state
                    
    #add_state = []
    for wrd in wrds :
        if wrd in st_short:
            return abv_to_state(wrd)
        else:
            if wrd in st_long:
                return wrd
    
    return        


fhc = codecs.open("../hc_train.txt", 'r', "utf-8")
fdt = codecs.open("../dt_train.txt", 'r', "utf-8")

pos = []
neg = []
neu = []

with fhc:
	for line in fhc:
                try:
                    lan = detect(line)
                    if lan == 'en':
                        t = line.split("\t")
                        if len(t) == 2:
                            tweet = filter_tw(t[0])
                            cat = t[1].strip()
                            if cat == '1':
                                pos.append(tweet)
                            if cat == '-1':
                                neg.append(tweet)
                            if cat == '0':
                                neu.append(tweet)
                except:
                    pass

#for i in range(10):
#    print('Round ', i + 1)

if len(sys.argv) == 2:
    if sys.argv[1] == 'validate':
        print("Hillary Clinton")
        train_test(pos, neg, neu)

poscut = int(len(pos))
neucut = int(len(neu))
negcut = int(len(neg))

train_data1 = pos[:poscut] + neg[:negcut] + neu[:neucut]
train_labels1 = ['pos'] * poscut + ['neg'] * negcut + ['neu'] * neucut

vectorizer1 = TfidfVectorizer(min_df=0, max_df = 0.9, \
                             sublinear_tf=True, use_idf=True)
train_vectors1 = vectorizer1.fit_transform(train_data1)

pos = []
neg = []
neu = []

with fdt:
	for line in fdt:
		t = line.split("\t")
		if len(t) == 2:
        		tweet = filter_tw(t[0])
        		cat = t[1].strip()

        		if cat == '1':
            			pos.append(tweet)
        		if cat == '-1':
            			neg.append(tweet)
        		if cat == '0':
            			neu.append(tweet)

if len(sys.argv) == 2:
    if sys.argv[1] == 'validate':
        print("Donald Trump")
        train_test(pos, neg, neu)
        sys.exit()

#for i in range(10):
#    print('Round ', i + 1)
#    train_test(pos, neg, neu)

poscut = int(len(pos))
neucut = int(len(neu))
negcut = int(len(neg))

train_data2 = pos[:poscut] + neg[:negcut] + neu[:neucut]
train_labels2 = ['pos'] * poscut + ['neg'] * negcut + ['neu'] * neucut

vectorizer2 = TfidfVectorizer(min_df=0, max_df = 0.9, \
                             sublinear_tf=True, use_idf=True)
train_vectors2 = vectorizer2.fit_transform(train_data2)


# Perform classification with SVM, kernel=rbf
c_val = 1.0
classifier_rbf1 = svm.LinearSVC(C = 5.0)
classifier_rbf1.fit(train_vectors1, train_labels1)
classifier_rbf2 = svm.LinearSVC(C = 1.0)
classifier_rbf2.fit(train_vectors2, train_labels2)

#m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49, \
#            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
#m.readshapefile('/home/prasad/Downloads/basemap-1.0.7/examples/st99_d00', \
#                name='states', drawbounds=True)

hc_st = []
dt_st = []
hc_wbg = []
dt_wbg = []
all_cnt = 0

db = pymysql.connect(host="10.180.55.14", user="twitter", \
	password="tw33t", db="pres_el16", charset = 'utf8mb4')
with db:
    cur = db.cursor()
    #cur.execute("select tweet, lat, lon from twitter;")
    cur.execute("select username, text from tweets;")
    tweets = cur.fetchall()

    cur.execute("select username, location from user_loc;")
    usr_loc = cur.fetchall()
    loc_dict = {}
    for item in usr_loc:
            usr = item[0]
            if item[1] is not None:
                if ' ' in item[1]:
                    loc = item[1].split()
                    loc_state = state_check(' '.join(loc))
                else:
                    loc = item[1]
                    loc_state = state_check(loc)
                #is it loc_state[0] or reject if len(loc_state) > 1) ?
                if loc_state == None:
                    print(item[1], ' '.join(loc))
                loc_dict[usr] = loc_state

    for usr_name, text in tweets:
        twt = text.lower()
        twt = filter_tw(twt)
        wrds = twt.split()
        
        if usr_name in loc_dict.keys():
            all_cnt = all_cnt + 1
            st = loc_dict[usr_name]
            if (('hillary' in wrds) or ('hilary' in wrds) or \
                ('clinton' in wrds) or ('hrc' in wrds)) and \
                not (('donald' in wrds) or ('trump' in wrds)):
                #st = get_state(float(tweet[2]), float(tweet[1]))
                pred = classifier_rbf1.predict(vectorizer1.transform([twt]))
                
                if pred[0] == 'neg':
                    #x, y = m(tweet[2], tweet[1])
                    #m.plot(x, y, 'o', color = 'blue', markersize=4)
                    hc_st.append(st)
                for wrd in wrds:
                    if wrd not in ['hillary', 'hilary', 'clinton', \
                                   'hrc', 'rt', 'httpst', 'will']:
                        hc_wbg.append(wrd)
            else:
                if (('donald' in wrds) or ('trump' in wrds)) and \
                   not (('hillary' in wrds) or ('hilary' in wrds) or \
                        ('clinton' in wrds) or ('hrc' in wrds)):
                    pred = classifier_rbf2.predict(vectorizer2.transform([twt]))
                    if pred[0] == 'neg':
                        #x, y = m(tweet[2], tweet[1])
                        #m.plot(x, y, 'o', color = 'red', markersize=4)
                        dt_st.append(st)
                        for wrd in wrds:
                            if wrd not in ['donald', 'trump', 'rt', \
                                           'httpst', 'will']:
                                dt_wbg.append(wrd)

print(len(hc_st) + len(dt_st))
print(all_cnt)

hc_d = Counter(hc_st)
dt_d = Counter(dt_st)

print('State-wise results for Hillary Clinton')
print(hc_d)
print('State-wise results for Donald Trumph')
print(dt_d)

#state_names = []
#for shape_dict in m.states_info:
#    state_names.append(shape_dict['NAME'])
#state_names = list(set(state_names))

#ax = plt.gca() # get current axes instance

#for s in hc_d.keys():
#    if s in dt_d.keys():
#        clr = 'white'
#        if hc_d[s] > dt_d[s]:
#            clr = 'red'
#        else:
#            if dt_d[s] > hc_d[s]:
#                clr = 'blue'
#        st = abv_to_state(s)
#        if st != 'NAS':
#            seg = m.states[state_names.index(st)]
#            poly = Polygon(seg, facecolor = clr, edgecolor = clr)
#            ax.add_patch(poly)
        
#plt.show()

#plot_wordcloud(hc_wbg, 'Clinton')
#plot_wordcloud(dt_wbg, 'Trump')

