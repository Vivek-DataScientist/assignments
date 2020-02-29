
import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 
from nltk.corpus import stopwords #importing stopwords
import matplotlib.pyplot as plt #importing plots
from wordcloud import WordCloud #importing Wordcloud
#extract Review from Amazon product Review
rev=[]
for i in range(1,200):
  iu=[]  
  url = "https://www.amazon.in/product-reviews/B01J82IYLW/ref=acr_dpproductdetail_text?ie=UTF8&showViewpoints=1"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})
  # Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    iu.append(reviews[i].text)  
  rev=rev+iu 
#join all reviews in single
ip1_rev_string = " ".join(rev)

# Removing unwanted symbols incase if exists
ip1_rev_string = re.sub("[^A-Za-z" "]+"," ",ip1_rev_string).lower()
ip1_rev_string = re.sub("[0-9" "]+"," ",ip1_rev_string)
#words separated for headset reviews
ip1_reviews_words = ip1_rev_string.split(" ")
#removing stop words
with open("K:\\Ds datas\\extractind review\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords = stopwords.split("\n")
ip_review_words=[e for e in ip1_reviews_words if not e in stopwords]
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_review_words)
# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)
#take +ve words
with open("K:\\Ds datas\\extractind review\\positive-words.txt","r")as pv:
    positive=pv.read().split('\n')
positive=positive[36:]
# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_review_words if w in positive])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
#take -ve words
with open("K:\\Ds datas\\extractind review\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_review_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)
