import os
import requests
from bs4 import BeautifulSoup

from PyPDF2 import PdfReader
import numpy as np

from urllib.parse import urljoin

folder_location = "data/berkshirehathaway"
if not os.path.exists(folder_location):
    os.mkdir(folder_location)

url = "https://www.berkshirehathaway.com/reports.html"
agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
}
response = requests.get(url, headers=agent)

soup = BeautifulSoup(response.text, "html.parser")
for link in soup.select("a[href$='.html']"):
    _url = "https://www.berkshirehathaway.com/" + link["href"]
    _response = requests.get(_url, headers=agent)
    _soup = BeautifulSoup(_response.text, "html.parser")
    for a in _soup.select("a[href$='.pdf']"):
        filename = os.path.join(folder_location, a["href"].split("/")[-1])
        with open(filename, "wb") as f:
            f.write(requests.get(urljoin(_url, a["href"]), headers=agent).content)

for link in soup.select("a[href$='.pdf']"):
    _url = "https://www.berkshirehathaway.com/" + link["href"]

    filename = os.path.join(folder_location, link["href"].split("/")[-1])
    with open(filename, "wb") as f:
        f.write(requests.get(_url, headers=agent).content)


directory = os.fsencode(folder_location)
q1 = "data/q1.txt"
q2 = "data/q2.txt"
q3 = "data/q3.txt"
q4 = "data/annualreport.txt"
other = "data/other.txt"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.startswith("1st"):
        reader = PdfReader(f"{folder_location}/{filename}")
        with open(q1, "a") as f:
            for page in range(len(reader.pages)):
                f.write(reader.pages[page].extract_text())
    elif filename.startswith("2nd"):
        reader = PdfReader(f"{folder_location}/{filename}")
        with open(q2, "a") as f:
            for page in range(len(reader.pages)):
                f.write(reader.pages[page].extract_text())
    elif filename.startswith("3rd"):
        reader = PdfReader(f"{folder_location}/{filename}")
        with open(q3, "a") as f:
            for page in range(len(reader.pages)):
                f.write(reader.pages[page].extract_text())
    elif filename.endswith("ar.pdf"):
        reader = PdfReader(f"{folder_location}/{filename}")
        with open(q4, "a") as f:
            for page in range(len(reader.pages)):
                f.write(reader.pages[page].extract_text())
    else:
        reader = PdfReader(f"{folder_location}/{filename}")
        with open(other, "a") as f:
            for page in range(len(reader.pages)):
                f.write(reader.pages[page].extract_text())


import tiktoken

enc = tiktoken.get_encoding("gpt2")

# file_path = os.path.join(os.path.dirname(__file__))
# files = [
#     file for file in os.listdir(os.path.dirname(__file__)) if file.endswith(".txt")
# ]
# print(files)


# train_bin = np.array([])
# val_bin = np.array([])
# for file in files:
#     with open(file, "r") as f:
#         data = f.read()
#     n = len(data)
#     train_data = data[: int(n * 0.9)]
#     val_data = data[int(n * 0.9) :]
#     train_ids = enc.encode_ordinary(train_data)
#     val_ids = enc.encode_ordinary(val_data)
#     print(f"train {file} has {len(train_ids):,} tokens")
#     print(f"val {file} has {len(val_ids):,} tokens")
#     train_bin = np.concatenate([train_bin, np.array(train_ids, dtype=np.uint16)])
#     val_bin = np.concatenate([val_bin, np.array(val_ids, dtype=np.uint16)])
# train_bin.tofile("train.bin")
# val_bin.tofile("val.bin")


files = [
    file for file in os.listdir(os.path.dirname(__file__)) if file.endswith(".txt")
]
print(files)

enc = tiktoken.get_encoding("gpt2")
train_bin = []
val_bin = []
for file in files:
    filepath = os.path.join(os.path.dirname(__file__), file)
    with open(filepath, "r") as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train {file} has {len(train_ids):,} tokens")
    print(f"val {file} has {len(val_ids):,} tokens")

    # train_bin.append(train_ids)
    # val_bin.append(val_ids)
    train_bin = train_bin + train_ids
    val_bin = val_bin + val_ids
    # train_bin = np.concatenate([train_bin, np.array(train_ids, dtype=np.uint16)])
    # val_bin = np.concatenate([val_bin, np.array(val_ids, dtype=np.uint16)])
# export to bin files
train_ids = np.array(train_bin, dtype=np.uint16)
val_ids = np.array(val_bin, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))