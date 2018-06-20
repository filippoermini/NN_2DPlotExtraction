import json
print("Progetto Python")
# importo il file json
with open('serialize7Line.json') as f:
    data = json.load(f)

print(data)