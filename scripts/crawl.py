from bs4 import BeautifulSoup
import requests
import time


url = "https://www.coffeereview.com/review/page/"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:106.0) Gecko/20100101 Firefox/106.0"}


r = requests.get(url, headers=headers)

texts = []
for i in range(1, 346):
    url = "https://www.coffeereview.com/review/page/" + str(i) + "/"
    r = requests.get(url, headers=headers)
    texts.append(r.text)
    # print status
    print("Page " + str(i) + " done")
    time.sleep(1)

print(len(texts))

# save the texts
with open("data/coffee_review.txt", "w") as f:
    for text in texts:
        f.write(text)
