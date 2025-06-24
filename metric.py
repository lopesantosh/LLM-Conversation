import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def get_cosine(text1, text2):
    vectorizer = TfidfVectorizer()
    texts = [text1, text2]
    tfidf = vectorizer.fit_transform(texts) # number of rows is len(texts)
    
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    sim = torch.from_numpy(sim)
    return sim[0][0]


if __name__ == "__main__":
    text1 = "I work for IBM"
    text2 = "IBM works in AI"
    cs = get_cosine(text1, text2)
    print(f"Cosine similarity: {cs:0.3f}")