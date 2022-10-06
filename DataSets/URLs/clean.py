import re
from urllib import parse

with open('datasets/URLs/badqueries.txt','r',encoding='utf-8') as f:
    data = f.readlines()
    cleaned_data = []
    for line in data:
        
        s = line.split(' //')
        if len(s) > 1:
            cleaned_data += s#[1].rstrip('\n')
    print(cleaned_data)

