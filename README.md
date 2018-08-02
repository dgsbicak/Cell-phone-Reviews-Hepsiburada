# Cell Phone Reviews from Hepsiburada.com
Around 44,000 customer reviews gathered by using Python's data mining libraries. 
Not every customer revealed their gender. That is why I used this data to assign genders based on customers' names:  
https://gist.github.com/ismailbaskin/1325813


# Preprocessing
```
# Create a Brand column
df['Brand'] = df['link'].apply(lambda x: x.split('-')[0][1:])
# Discard brands with less than 20 comments
a = df['Brand'].value_counts()>20
blst=list()
for n in range(0,len(a)):
    if a[n]:
        blst.append(a.index[n])
df = df[df.loc[:,'Brand'].map(lambda x: x in blst)]
df.loc[:,'Brand'].value_counts()
```

#### Create Gender column
```
def simple_gender(x):
    if isinstance(x, float):
        #print('ITS FLOAT! NAN',x)
        return np.nan
    x = x.split(' ')[0].lower()
    m = pd.read_csv('erkekisimleri.csv').iloc[:,1].tolist()
    f = pd.read_csv('kadinisimleri.csv').iloc[:,1].tolist()
    u = pd.read_csv('unisexisimler.csv').iloc[:,1].tolist()
    if (x == 'erkek') or (x in m):
        return 'M'
    elif (x == 'kadın') or (x in f):
        return 'F'
    elif x in u:
        return 'U'
    else:
        #print("PROBLEM OCCURED WHILE ASSIGNING GENDERS!",x)
        return np.nan
df['Gender'] = df['Name'].apply(simple_gender)
df['Gender'].value_counts()
```

#### Age Text Cleaning
```
def num_cleaner2(x):
    cln=list()

    if isinstance(x, str):
        #print(x)
        if '.' in list(x):
            y = ''.join(list(x)[:-2])
            return int(y)
        for a in list(x):
            if a in [str(i) for i in range(0,10)]:
                cln.append(a)
            else:pass
        return int(''.join(cln))
    else:
        return np.nan
        #print(type(x))
# NaNs are floats
# floats appeared to be strings!
df['Age'] = pd.to_numeric(df['Age'].map(num_cleaner2))

def zeroage(x):
    cln = list()
    if x<9:  # Smaller than 9 year old wouldn't leave a comment
        print(x, "found a zero")
        return np.nan
    else:
        return x
df['Age'] = df['Age'].map(zeroage)
```

#### Create a Model column
```
df['Model'] = df['link'].apply(lambda x: x.split('-')[1]+"-"+x.split('-')[2])
```
#### Adjust the 'Location' column
```
def location_adjustment(x):
    if isinstance(x, float):
        pass
    else:
        x = x.replace('-',' ').replace('Türkiye','')
        return x
df['Location'] = df['Location'].apply(location_adjustment)
```
#### Adjust the 'Gender' column
```
def gender_sort(x):
    if isinstance(x, float) or x=='U':
        return 'Unknown'
    elif x=='M':
        return 'Male'
    elif x=='F':
        return 'Female'
    else:
        print('Something went wrong in gender_sort:',x)
        return x
df['Gender'] = df['Gender'].apply(gender_sort)
```
#### Clean the comments from stopwords and punctuations
```
def text_clean(text):
    stext = [word for word in text.split() if word.lower() not in stopwords.words('turkish')]
    ftext = ' '.join(stext)
    
    stext = [word for word in ftext if word not in string.punctuation]
    ftext = ''.join(stext)
    return ftext
t0 = time.time()
df['Comment'] = df['Comment'].apply(text_clean)
print("Completed within %0.1f seconds." % (time.time() - t0))
```
#### Clean remaining date figures.
```
def date6_delete(x):
    comment = list()
    for line in x.split():
        if (len(line)==8) and ('20' in line):
            pass
        else:
            comment.append(line)
    return " ".join(comment)
df['Comment'] = df['Comment'].apply(date6_delete)
```
#### Create a combined text ready for CountVectorizer, Tfidf
```
raw['Text'] = raw['Location'].fillna('')+' '+raw['CommentTitle'].fillna('')+' '+raw['Comment'].fillna('') \
        +' '+raw['Brand'].fillna('')+' '+raw['Model'].fillna('')+' '+raw['Gender'].fillna('')
```

# Descriptive Data Analysis 
#### Age Distribution
```
plt.subplots(figsize=(16,8))
sns.distplot(df['Age'].dropna(), bins=100)
plt.savefig('agedist1.png')
```
![agedist1](https://user-images.githubusercontent.com/23128332/41202223-84eb53e4-6cce-11e8-8f5c-d97ec50eb828.png)

#### Reviewer Gender Difference
```
#http://seaborn.pydata.org/tutorial/color_palettes.html
plt.subplots(figsize=(10,10))
sns.countplot(df1['Gender'].dropna(axis=0), palette=sns.xkcd_palette(['windows blue', 'greyish', 'pink']))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor((0.99,0.99,0.99))
plt.savefig('genderplot1.png')
```
Genders were predicted according to reviewers' names. Some names doesn't have different gender attributes on them. They are indicated as 'Unisex' or in other words, unknown gender information.
![genderplot1](https://user-images.githubusercontent.com/23128332/41202327-4adc5e76-6cd0-11e8-9555-7a7c6f2362f7.png)

#### Brand Ratings According to Genders
```
plt.subplots(figsize=(10,10))
sns.heatmap(pvt, cmap='magma', linecolor='white', linewidths=1)
plt.savefig('heatgenderrating2.png')
```
![heatgenderrating2](https://user-images.githubusercontent.com/23128332/41202224-850c9e82-6cce-11e8-9a94-268f44ef1f04.png)

#### Brand Ratings 
```
plt.subplots(figsize=(16,8))
plt.xticks(rotation=45)
sns.barplot(x=df['Brand'], y=df['Rating'], data=df)
plt.savefig('barplot1.png')
```
![barplot1](https://user-images.githubusercontent.com/23128332/41202328-4af929ca-6cd0-11e8-89f0-a0647d432064.png)

#### Pairplots
```
sns.pairplot(df.dropna(axis=0), hue='Gender', palette='rainbow')
plt.savefig('pairplot1.png')
```
![pairplot1](https://user-images.githubusercontent.com/23128332/41202326-4abe8270-6cd0-11e8-8e0f-62a8c6b7dfa8.png)


#### Apple Age Segment Distribution
```
fig, ax = plt.subplots()
sns.set(style="whitegrid")
sns.factorplot(x="Age_Segment", data=dage[dage['Brand']=='apple'], kind="count",
                   palette="rainbow", size=6, aspect=1.5,
                   order=["12–17","18–24","25–34","35–44","45–54","55–64","65+"],
              ax=ax)
fig.set_size_inches(16,8)
fig.savefig('Apple_AS1')
```
![apple_as1](https://user-images.githubusercontent.com/23128332/41435374-47f27254-7027-11e8-9ee1-e91c6b4d7ec3.png)

#### Samsung Age Segment Distribution
```
fig, ax = plt.subplots()
sns.set(style="whitegrid")
sns.factorplot(x="Age_Segment", data=dage[dage['Brand']=='samsung'], kind="count",
                   palette="rainbow", size=6, aspect=1.5,
                   order=["12–17","18–24","25–34","35–44","45–54","55–64","65+"],
              ax=ax)
fig.set_size_inches(16,8)
fig.savefig('Samsung_AS1')
```
![samsung_as1](https://user-images.githubusercontent.com/23128332/41435380-4ab2e51e-7027-11e8-91f7-3cdf7bf9743b.png)

#### Unfamous Brands' Age Segmet Distribution

```
g = sns.factorplot(x="Age_Segment", data=dage[dage['Brand']=='other'], kind="count",
                   palette="rainbow", size=6, aspect=1.5,
                   order=["12–17","18–24","25–34","35–44","45–54","55–64","65+"])
plt.title('Other')
```
![resim](https://user-images.githubusercontent.com/23128332/43580995-f95614d4-965f-11e8-88c2-0115f31942d5.png)
If we assume that every comment indicates the customers who had bought the products, and had left a comment on it afterwards, we can hypothesise the following conclusions:
People who are in 18-24 and 25-34 age segments more likely to prefer Apple mobile phones instead of buying Samsung mobile phones.
And people who are in 35-44 age segment are more likely to prefer Samsung mobile phones instead of Apple.
**But, we need to make statistical tests to be more certain.**



# Vectorization
### CountVectorizer
```
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
feature_train_counts = count_vec.fit(raw['Text'])
bag_of_words = feature_train_counts.transform(raw['Text'])
from sklearn.model_selection import train_test_split
X = bag_of_words 
y = raw['Rating'].fillna('')
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.33, shuffle=True)
```
# Model Building and Prediction
#### Stocastic Gradient Descent Classifier
```
from sklearn.naive_bayes import SGDClassifier
clf = SGDClassifier().fit(feature_train, label_train)
preds_sgd = clf.predict(feature_test)
from sklearn.metrics import confusion_matrix, classification_report
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(label_test, preds_sgd))))
print(confusion_matrix(label_test, preds_sgd))
print('\n')
print(classification_report(label_test, preds_sgd))
```
