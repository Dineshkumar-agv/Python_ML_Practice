
import requests 
from bs4 import BeautifulSoup

# headers = {
#     'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }

# login_data = {
#     'name': '<username>',
#     'pass': '<password>',
#     'form_id': 'new_login_form',
#     'op': 'Login'
# }

with requests.Session() as s:
    url = 'https://edwisor.com/'
    r = s.get(url)
    # soup = BeautifulSoup(r.content, 'html5lib')
    # login_data['form_build_id'] = soup.find('input', attrs={'name': 'form_build_id'})['value']
    # r = s.post(url, data=login_data, headers=headers)
    # print(r.content)