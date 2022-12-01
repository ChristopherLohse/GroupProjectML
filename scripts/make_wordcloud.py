
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
df = pd.read_csv('data/coffee_review.csv')


df["text"] = df["text"].astype(str)


# rename the nan values
df.dropna(inplace=True)
# remove duplicates
df.drop_duplicates(inplace=True)


# remove stop words
stop_words = set(stopwords.words('english'))
df["text"] = df["text"].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop_words)]))


# wordcloud with the text
wordcloud = WordCloud(background_color="white",
                      collocations=True).generate(" ".join(df["text"], ))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# save the wordcloud tight
plt.savefig("plots/cloud.png", dpi=300, bbox_inches='tight')
