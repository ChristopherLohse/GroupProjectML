
import bs4 as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# set style tp seaborn
plt.style.use('seaborn-whitegrid')
# read the file
with open("data/coffee_review.txt", "r") as f:
    texts = f.read()

# parse the texts
soup = bs.BeautifulSoup(texts)


# filter everything that is class="review-template "
reviews = soup.find_all("div", class_="review-template")


# iterate over the reviews
ratings = []
texts = []
titles = []
prices = []
review_dates = []
for review in reviews:
    # get the value for class=review-template-rating
    rating = review.find(class_="column col-1").text.strip().replace("\n", "")
    try:
        rating = int(rating)
    except:
        rating = np.nan
    ratings.append(rating)
    title = review.find(class_="review-title").text.strip().replace("\n", "")
    titles.append(title)
    text = review.find(class_="row row-2").text.strip().replace("\n", "")
    texts.append(text)
    price_review = review.find(class_="column col-3").text
    # split on /n and take the second element
    price = price_review.split("\n")[2].strip().replace(
        "\n", "").replace("Price: NT ", "")
    prices.append(price)
    review_date = price_review.split("\n")[1].strip().replace(
        "\n", "").replace("Review Date:", "")
    review_dates.append(review_date)


df = pd.DataFrame({"rating": ratings, "title": titles,
                  "text": texts, "price": prices, "review_date": review_dates})


df.head()


# count nan values
print(df.isna().sum())


# get the range for the rating
print(df["rating"].min(), df["rating"].max())
df.dropna(inplace=True)


# remove null values from the text
df = df[df['text'].notnull()]

# boxplot of the ratings save the figure
plt.boxplot(df["rating"])
# change the y axis label
plt.ylabel("Rating")
# remove x axis label and ticks
plt.xlabel("")
plt.xticks([])
plt.savefig("plots/boxplot.png", dpi=300, bbox_inches='tight')

# save the dataframe
df.to_csv("data/coffee_review.csv", index=False)

print("saved the dataframe")
