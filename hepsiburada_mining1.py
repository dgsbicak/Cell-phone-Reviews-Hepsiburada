from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pickle
import os
import csv
import pandas as pd
import time
import random

# Some Pickle Functions
def save_pickle(obj, fname):
    try:os.mkdir('obj/')
    except:pass
    with open('obj/'+fname + '.pk', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
    print("Object saved as a pickle: {}".format(obj))
def load_pickle(fname):
    with open('obj/' + fname + '.pk', 'rb') as f:
        return pickle.load(f)
    
def page_finder(bs):
    try:
        pages = bs.find('div', class_="pagination").findAll('a')
        page_last = str(list(pages)[-1]).split('sayfa=')
        last_number = page_last[-1].split('">')[0].split('&')[0]
        return int(last_number)
    except:
        return 1
def comment_write(bag, phoneID, pagenu):
    try:os.mkdir('comments/')
    except:pass
    df = pd.DataFrame.from_dict(bag, orient='index').T.to_csv('comments/'+phoneID+'_'+pagenu+'.csv',index=False)
    print('CSV IS WRITTEN: {}_{}.csv'.format(phoneID,pagenu))

def hepsib_status1(pagenum, phone_links):
    print('{}th Phone batch that is about to be examined:'.format(pagenum))
    for phone in phone_links:
        print(phone)
    # Save the status on a text file.
    with open('Status.txt', 'w') as f:
        f.write('{}th Phone batch is being examined'.format(pagenum))
        f.close()

# Main Phone Catalog Page
url = 'https://www.hepsiburada.com/cep-telefonlari-c-371965?siralama=coksatan&sayfa='
html = urlopen(url)
bs = BeautifulSoup(html, 'html.parser')

# Find how many pages there are in the Phone Catalog Page
last_number= page_finder(bs)

for page_num in range(1,last_number+1):
    if not os.path.isfile('obj/links_sayfa' + str(page_num) + '.pk'):
        url = 'https://www.hepsiburada.com/cep-telefonlari-c-371965?siralama=coksatan&sayfa='+ str(page_num)
        html = urlopen(url)
        bs = BeautifulSoup(html, 'html.parser')

        catalog = bs.find_all('a', {'data-bind':'click: clickHandler.bind($data)'})
        hrefs=list()
        for cat in catalog:

            a = re.search('href=(.*)"', str(cat)) # Get the links
            if a:
                cleaned = a.group()[5:].replace("\"", "") # Cleaned Links
                hrefs.append(cleaned)
            else:
                print('No match!')

        # Save the Phone Links into a Pickle
        if len(hrefs)>1:
            save_pickle(hrefs, 'links_sayfa'+ str(page_num))
        else:
            print('No links in the list')
    else:
        print('links_sayfa'+str(page_num)+' already exists.')

# Get into the Comment Section

for pagenum in range(1,last_number+1): # continue from the page 3
    phone_links = load_pickle('links_sayfa'+str(pagenum))
    hepsib_status1(pagenum, phone_links) # Save the current status
    com_PNu = 0
    # Comment Main Page
    for link in phone_links:
        ID=link.split('-')[-2:]
        if ID[0]=='p':
            phoneID=ID[1]
        else:
            # Solution for duplicate csv names
            phoneID="{}-{}".format(ID[0],ID[1])
        
        print('Starting to collect the comments for {}'.format(phoneID))
        url = 'https://www.hepsiburada.com/' + link + '-yorumlari'
        html = urlopen(url)
        bs = BeautifulSoup(html, 'html.parser')
        # How many pages of Comments there are for the phone?
        com_PNu = page_finder(bs)
        for page in range(1, com_PNu+1):
            url = 'https://www.hepsiburada.com/' + link + '-yorumlari?sayfa=' + str(page)
            html = urlopen(url)
            bs = BeautifulSoup(html, 'html.parser')

            # Save the Comment Info
            titles = bs.find_all('strong', 'subject')
            comments = bs.find_all('p', 'review-text')
            dates = bs.find_all('strong', 'date')
            names = bs.find_all('span', 'user-info')
            ratings = bs.find_all('div', 'ratings active')
            loc = bs.find_all('span', 'location')
            phelp = bs.find_all('a', {'data-agreed':'true'})
            nhelp = bs.find_all('a', {'data-agreed':'false'})

            # Dates
            datelist = list()
            for date in dates:
                date = str(date)[-20:-10]
                datelist.append(date)

            # Names & Ages
            namelist = list()
            agelist = list()
            for name in names:
                name = str(name).replace('<span class=\"user-info\">', '').replace('</span>','')
                if len(name)==0:
                    namelist.append('')
                    agelist.append('')
                elif '(' in list(name):
                    age = name[-3:-1]
                    name = name[:-5]
                    namelist.append(name)
                    agelist.append(age)
                else:
                    namelist.append(name)
                    agelist.append('')

            # Locations
            loclist = list()
            for location in loc:
                location = str(location).replace('<span class="location">', '').replace('</span>','')
                if len(location.split(' - '))==1:
                    loclist.append('')
                else:
                    loclist.append(location)

            # CommentTitle
            titlelist = list()
            for title in titles:
                title = str(title).replace('<strong class="subject">', '').replace('</strong>', '')
                titlelist.append(title)

            # Comment
            commentlist = list()
            for comment in comments:
                comment = str(comment).replace('<p class="review-text">', '').replace('</p>', '')
                commentlist.append(comment)

            # Rate
            ratelist = list()
            for rate in list(ratings)[6:]:
                rate = str(rate).replace('<div class="ratings active" style="width: ', '')
                ratelist.append(rate.split('%')[0])

            # p_helpful
            plist = list()
            for p in phelp:
                pos = str(p).split(' <b>(')[1]
                pos = pos.split(')</b></a>')[0]
                plist.append(pos)

            # n_helpful
            nlist = list()
            for n in nhelp:
                neg = str(n).split(' <b>(')[1]
                neg = neg.split(')</b></a>')[0]
                nlist.append(neg)
            comments = {'Date':datelist, 'Name':namelist,'Age':agelist,'Location':loclist,'CommentTitle':titlelist,
                        'Comment':commentlist,'Rating':ratelist,'p_helpful':plist,'n_helpful':nlist, 'link':[link]*len(nlist)}
            comment_write(comments, phoneID, str(page))
            print('Waiting for 10 secs.')
            time.sleep(10)

