#import package
import pandas as pd#pandas module
book=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\reccomdation\\bookse.csv",encoding = "ISO-8859-1")

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
# Creating a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words="english") 

#fill empty as ""
book["Book.Title"].isnull().sum() #No Null Values
book["Book.Author"].isnull().sum() #No Null Values
book["Publisher"].isnull().sum() #No Null Values
#if empty filled with ""
# Preparing the Tfidf matrix by fitting and transforming

#change column names
book.rename(columns={'Book.Author':'author'},inplace=True)
tfidf_matrix = tfidf.fit_transform(book.author)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape  #  (5000, 8438)

#importing linear kernel
from sklearn.metrics.pairwise import linear_kernel
# Computing the cosine similarity on Tfidf matrix
cosine1_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

book_index = pd.Series(book.index,index=book['Book.Title']).drop_duplicates()

#function 
def get_book_recommendations(name,topN):
    #topN = 10 top 10 book
    # Getting the book index using its title 
    book_id = book_index[name]
    # Getting the pair wise similarity score for all the books with that 
    cosine_scores1 = list(enumerate(cosine1_sim_matrix[book_id]))
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores1 = sorted(cosine_scores1,key=lambda x:x[1],reverse = True)
    # Get the scores of top 10 most similar books 
    cosine_scores_10 = cosine_scores1[0:topN+1]
    # Getting the book index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    # Similar books and scores
    book_similar_show = pd.DataFrame(columns=["Book.Title","Publisher","ratings[,3]"])
    book_similar_show["Book.Title"] = book.loc[book_idx,"Book.Title"]
    book_similar_show["Publisher"] = book.loc[book_idx,"Publisher"]
    book_similar_show["ratings[, 3]"] = book.loc[book_idx,"ratings[, 3]"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print (book_similar_show)
    #return (book_similar_show)

# Enter your book and number of books to be recommended by particular author
get_book_recommendations("The Martian Chronicles",topN=10)

#get book recommendation based on author
#recommended book with publisher and ratings