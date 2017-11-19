from pytrends.request import TrendReq
import csv
import json

google_username = ""	# valid gmail id
google_password = ""	# password

pytrends = TrendReq(google_username, google_password, custom_useragent=None)

payload = {'geo': 'US'}

hot = pytrends.hottrends(payload)
print json.dumps(hot)

f = open("hot.txt", "w+")
f.write(json.dumps(hot).encode('utf-8'))
f.close()