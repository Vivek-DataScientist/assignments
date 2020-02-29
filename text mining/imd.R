library(rvest)
library(magrittr)
library(XML)
a<-10
joker<-NULL
urll<-"https://www.imdb.com/title/tt7286456/reviews?ref_=tt_urv"
for(i in 1:30){
  url<-read_html(as.character(paste(urll,i*a,sep="")))
  pingg<-url %>%
    html_nodes(".show-more__control") %>%
    html_text() 
  joker<-c(joker,pingg)
}
joker#get all reviews of movie joker
write.csv(joker,file="joker.csv")
write.table(joker,file="joker.txt",row.names = FALSE)

#install.packages("tm") #install tm package
library(tm) #import tm package
#clean dataset
imd_corpus <- Corpus(VectorSource(joker))
imd_corpus <- tm_map(sms_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

# clean up the corpus using tm_map()
corpus_clean <- tm_map(imd_corpus, tolower) #change to lower
corpus_clean <- tm_map(corpus_clean, removeNumbers) #remove numbers
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords()) #remove stopwords
corpus_clean <- tm_map(corpus_clean, removePunctuation) #remove punctuation
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

######### Emotion Mining ###############
library(syuzhet)
library(plotly)
library(tm)
s_v <- get_sentences(joker)

syuzhet <- get_sentiment(s_v, method="syuzhet")
bing <- get_sentiment(s_v, method="bing")
afinn <- get_sentiment(s_v, method="afinn")
nrc <- get_sentiment(s_v, method="nrc")
sentiments <- data.frame(syuzhet, bing, afinn, nrc)

#anger", "anticipation", "disgust", "fear", "joy", "sadness", 
#"surprise", "trust", "negative", "positive."
emotions <- get_nrc_sentiment(joker)
head(emotions)
emo_bar = colSums(emotions)
barplot(emo_bar)
emo_sum = data.frame(count=emo_bar, emotion=names(emo_bar))

plot(syuzhet, type = "l", main = "Plot Trajectory",
     xlab = "Narrative Time", ylab = "Emotional Valence")

# To extract the sentence with the most negative emotional valence
negative <- s_v[which.min(syuzhet)]
negative

#negative words
# "Even if taken into account that the poverty and anger was cooking 
# for a long time in that society, it still feels somewhat absurd that 
# killing three people in the subway made the joker so notorious."

# and to extract the most positive sentence
positive <- s_v[which.max(syuzhet)]
positive

#positive words
# "Philips on the other hand had the perfect choice of music,
#  camera angles, lighting and every important factor."


#Create document-term matrix
dtm <- DocumentTermMatrix(imd_corpus)
#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)# 1101

#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
freq[ord]
