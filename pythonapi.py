import http.client
def callapi(data):
    conn = http.client.HTTPSConnection("neutrinoapi-bad-word-filter.p.rapidapi.com")
    data=data.split()
    stack=""
    i=0
    for abc in data:
        if i == 0:
            stack=stack+abc
            i=i+1
        stack=stack+"%20"+abc
    
    payload = "censor-character=*&content="
    payload = payload+stack
    
    headers = {
        'x-rapidapi-host': "neutrinoapi-bad-word-filter.p.rapidapi.com",
        'x-rapidapi-key': "89de85434emshc54447497b32483p178e75jsne5fcef5fd57c",
        'content-type': "application/x-www-form-urlencoded"
    }

    conn.request("POST", "/bad-word-filter", payload, headers)

    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")