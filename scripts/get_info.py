import requests
import json

url = "https://api.nusmods.com/v2/2024-2025/modules/CS3263.json"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    with open("CS3263.json","w") as f:
        json.dump(data,f)
        print("Saved")
else:
    print("Request failed with status code: ", response.status_code)