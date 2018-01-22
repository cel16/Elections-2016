import csv
import re
import sys

tags = {}
mntn = {}

for i in [31] + range(1,9):
    with open('dt_'+str(i)+'.csv', 'r') as csvfile:
        #reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL, quotechar='"')
        #for row in reader:
        for line in csvfile:
            #tweet = ' '.join(row)
            #            if "@" in tweet or "#" in tweet:
            tweet = line
            for m in set(re.findall(r'@\w+', tweet)):
                if m in mntn.keys():
                    mntn[m] = mntn[m] + 1
                else:
                    mntn[m] = 1
            for t in set(re.findall(r'#\w+', tweet)):
                if t in tags.keys():
                    tags[t] = tags[t] + 1
                else:
                    tags[t] = 1

sort_tags = [(v, k) for k,v in tags.iteritems()]
sort_tags.sort(reverse=True)
for item in sort_tags:
    print item
#for i in range(20):
#    print temp[i]

sort_mntn = [(v, k) for k,v in mntn.iteritems()]
sort_mntn.sort(reverse=True)
for item in sort_mntn:
    print item
#for i in range(20):
#    print temp[i]

#print sorted(((v, k) for k,v in tags.iteritems()), reverse=True)
#print sorted(((v, k) for k,v in mntn.iteritems()), reverse=True)
