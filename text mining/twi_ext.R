library("twitteR")
library("ROAuth")
cred <- OAuthFactory$new(consumerKey='N1ULs6gmyRUiGWjCHvO6xOQGo', # Consumer Key (API Key)
                         consumerSecret='g4vM12p5osTyC7c5gEwM2eXCIqvRBqlLFJX24aR8ddFjBnqU2Q', #Consumer Secret (API Secret)
                         requestURL='https://api.twitter.com/oauth/request_token',
                         accessURL='https://api.twitter.com/oauth/access_token',
                         authURL='https://api.twitter.com/oauth/authorize')
save(cred, file="twitter authentication.Rdata")
load("twitter authentication.Rdata")
#install.packages("base64enc")
library(base64enc)
#install.packages("httpuv")
library(httpuv)
setup_twitter_oauth("N1ULs6gmyRUiGWjCHvO6xOQGo",
                    "g4vM12p5osTyC7c5gEwM2eXCIqvRBqlLFJX24aR8ddFjBnqU2Q",
                    "2522296758-MDdw5ROs9B9FPJBGjYhEWRPgb3Rd6jbxIiiuy68",
                    "BKeJogscMNtqyT4p4Qf4VPl6mERsb5QPhhTBNm1qBgTlQ")
registerTwitterOAuth(cred)
#extract tweets of cricketer Raina
Tweets <- userTimeline('ImRaina', n = 100,includeRts = T)
TweetsDF <- twListToDF(Tweets)
dim(TweetsDF)
View(TweetsDF)
Tweets <- paste(Tweets, collapse = " ")
############ Topic extraction ##################
library(tm)
library(slam)
library(topicmodels)
#install.packages("wordcloud2")
library(wordcloud)
library(wordcloud2)

#cleaning the data
mydata.corpus <- Corpus(VectorSource(TweetsDF$text))
mydata.corpus <- tm_map(mydata.corpus,tolower)
mydata.corpus <- tm_map(mydata.corpus, removePunctuation) 
my_stopwords <- c(stopwords('english'),"the", "due", "are", "not", "for", "this", "and",  "that", "there", "new", "near", "beyond","will","also","can", "time", 
                  "from", "been", "both", "than",  "has","now", "until", "all", "use", "two", "ave", "blvd", "east", "between", "end", "have", "avenue", "before",    "just", "mac", "being",  "when","levels","remaining","based", "still", "off", 
                  "over", "only", "north", "past", "twin", "while","then"
                  ,"Brothers","Sisters","sisters")
mydata.corpus <- tm_map(mydata.corpus, removeWords, my_stopwords)
mydata.corpus <- tm_map(mydata.corpus, removeNumbers) 
mydata.corpus<- tm_map(mydata.corpus,stripWhitespace)
View(mydata.corpus)

library(syuzhet)
library(plotly)
library(tm)
s_v <- get_sentences(TweetsDF$text)
syuzhet <- get_sentiment(s_v, method="syuzhet")
bing <- get_sentiment(s_v, method="bing")
afinn <- get_sentiment(s_v, method="afinn")
nrc <- get_sentiment(s_v, method="nrc")
sentiments <- data.frame(syuzhet, bing, afinn, nrc)
emotions <- get_nrc_sentiment(TweetsDF$text)
head(emotions)
emo_bar = colSums(emotions)
barplot(emo_bar)
emo_sum = data.frame(count=emo_bar, emotion=names(emo_bar))

# To extract the sentence with the most negative emotional valence
negative <- s_v[which.min(syuzhet)]
negative
#Negative word in his Twitter :  "So terrible to see the hardships Australia is currently going though."
positive<- s_v[which.max(syuzhet)]
positive
#positive word in his twitter: "Role of youth is crucial in localising the #SDG &amp; should be encouraged by providing
#                              them right opportunities, platforms &amp; ment."

#Create document-term matrix
dtm <- DocumentTermMatrix(mydata.corpus)

#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)#585

#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
freq[ord]

library(NLP)
#document term matrix
lda <- LDA(dtm, 10) # find 10 topics
term <- terms(lda, 5) # first 5 terms of every topic

tops <- terms(lda)
tb <- table(names(tops), unlist(tops))
tb <- as.data.frame.matrix(tb)

cls <- hclust(dist(tb), method = 'ward.D2')
plot(cls)
#Cluster based on topics