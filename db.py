
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote

import bs4
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as etree

from pandas import DataFrame
from sklearn import tree
import lxml

url = 'http://apis.data.go.kr/1613000/BldRgstService_v2/getBrTitleInfo'
queryParams = '?' + \
              urlencode({ quote_plus('ServiceKey') : 'RAZnJDa8KOWGcokLV1Btdp5xtWOovGKIQr/TxrRvIP6LP6AEP3ud1WlDFHS7caIcHgx9U1G5K2DogPcBTCLyJA==',
                          quote_plus('sigunguCd') : '11680',
                          quote_plus('bjdongCd') : '10300',
                          # quote_plus('platGbCd') : '',
                          quote_plus('bun') : '',
                          quote_plus('ji') : '',
                          # quote_plus('startDate') : '20200101',
                          # quote_plus('endDate') : '20201231',
                          quote_plus('numOfRows') : '3',
                          # quote_plus('pageNo') : ''
                          })

request = Request(url + queryParams)
print(url+queryParams)
request.get_method = lambda: 'GET'
print(request)
response_body = urlopen(request).read()
# print(response_body)
output = response_body.decode('UTF-8')
# print(output)
soup = bs4.BeautifulSoup(response_body, features="html.parser")
data = soup.find_all('item')
print(data)
rows = []
for item in data:
    # print('제목 : ' + item.bldnm.get_text())
    # print('설명 : ' + item.newplatplc.get_text())

    cityname = item.bldnm.get_text()
    cityname2 = item.newplatplc.get_text()
    if item.vlRat is not None:
        cityname3 = item.vlrat.get_integer()
    else:
        cityname3 = "None"

    rows.append({"빌딩명": cityname, "도로명주소": cityname2, "용적률": cityname3})
    print(rows)

# print(rows)
columns = ["빌딩명", "도로명주소", "용적률"]
print(columns)
catalog_cd_df = pd.DataFrame(rows, columns = columns)
print(catalog_cd_df)

# soup2 = bs4.BeautifulSoup(response_body, features="lxml-xml")
# # print(soup2)
# data2 = soup.find_all('item')
# for item in data2:
#     print(item.bldnm.get_text())
#     print(item.find('bldnm'))
#     if item.vlRat is not None:
#         print(item.find('vlRat'))
#     else:
#         continue


# data2 = etree.ElementTree(response_body)
# data3 = data2.getroot()
# print(data3)
# rows = []
#
# for item in data3:
#     # print(object.find("bldnm").findtext("xmin"))
#     cityname = item.find('bldNm')
#     cityname2 = item.find('newplatplc')
#     print(cityname)
#     rows.append({"bldNm": cityname})
#     rows.append({"newPlatPlc": cityname2})
#
# columns = ["bldNm", "newPlatPlc"]
# catalog_cd_df = pd.DataFrame(rows, columns = columns)
# catalog_cd_df

# root = data.getroot()
# columns = ["bldnm", "newplatplc"]
# datatframe = pd.DataFrame(columns = columns)
# for node in root:
#     name = node.attrib.get("bldnm")
#     mail = node.find("newplatplc").text if node is not None else None
#     datatframe = datatframe.append(pd.Series([name, mail], index = columns), ignore_index = True)
#
# dataframe

# rows = []
# for item in data:
#      cityname = item.find('bldnm').text
#      cityname2 = item.find('newplatplc').text
#
#      # print(cityname.get_text())
#
#      rows.append({"bldnm": cityname})
#      rows.append({"newplatplc": cityname2})
#
# columns = ["bldnm", "newplatplc"]
# catalog_cd_df = pd.DataFrame(rows, columns = columns)

