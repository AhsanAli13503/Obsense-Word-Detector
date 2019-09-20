import json

json_string = '{"BTC":0.1434,"USD":387.92,"EUR":343.51}'
parsed_json = json.loads(json_string)

print (parsed_json['BTC'] )
# 0.1434

print (parsed_json['EUR'])
# 343.51