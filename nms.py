from urllib.request import urlopen
from bs4 import BeautifulSoup


html = urlopen("http://www.naver.com")
output = BeautifulSoup(html, "html.parser")

print(output)