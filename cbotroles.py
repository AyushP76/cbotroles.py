import nltk
import warnings

warnings.filterwarnings("ignore")
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wikipedia import page
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplit


sent_tokens= []
word_tokens= []

GREETING_INPUTS = ("hello", "yo", "hi", "greetings", "sup", "what's up", "wassup", "hey", "namaste")
GREETING_RESPONSES = ["hi", "nice to meet you..", "hey", "*nods*", "hi there", "hello", 'yes..?', 'greetings..']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Sorry, no data available for this."
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def main():
    flag=True
    print("\nHi, Enter the role you are asking for. Type 'Bye' to exit. Type 'New' to start new topic queries.")
    print("******************************************************************")
    topic=str(input("Please enter the topic you want to ask queries for: "))
    print("\nCollecting data...")
    print("******************")

    
    try:
        raw=page(topic)
        raw=raw.content
        raw=raw.lower()

    except:
        print("Sorry no data available for: ",topic + ". Try Again!!!")
        main()

    global sent_tokens
    sent_tokens = nltk.sent_tokenize(raw)

    global word_tokens
    word_tokens = nltk.word_tokenize(raw)

    print("Data Fetched. Ready to go...")

    while(flag==True):
        user_response = input("You: ")
        user_response=user_response.lower().strip()
        if(user_response!='bye' and user_response!='new'):
            if(user_response in ['thanks', 'thank you', 'thanx', 'thnx']):
                flag=False
                print("Cbot: You are welcome.")
            else:
                if(greeting(user_response)!=None):
                    print("Cbot: "+greeting(user_response))
                else:
                    print("Cbot: ",end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)

        elif(user_response=='new'):
            main()

        else:
            flag=False
            print("Cbot: Bye!!! See you soon.. :) ")
            break

if __name__ == '__main__':
    main()
