import requests

r = requests.get(r'http://www.reddit.com/user/spilcm/posts/.json',headers = {'User-agent':'your bot 0.1'})

# r.text

data = r.json()
# print data['data']
print data['data']
# for child in data['data']['children']:
#     print child['data']['id'], "", child['data']['author'],child['data']['body'],child['data']['score']
#     print
for child in data['data']['children']:
    print child['data']['id'], "", child['data']['author'],child['data']['score']
    print
