import json
from PIL import Image

print("Progetto Python")
# importo il file json
with open('serialize7Line.json') as f:
    data = json.load(f)
numLine = 7
imagePath = "/Users/filippo.ermini/Downloads/sample_train1/png/"
# ciclo per ogni immagine ed estraggo i pixel del bounding box
for imageSample in data:
    # mi creo l'array dei bounding box
    # ogni sample rappresenta i dati di una immagine
    index = imageSample['index']
    print("---------------------------------")
    print("Elaboro l'immagine numero "+`index`)
    numImages = len(imageSample['sampleArray'])
    i = 0
    bbArray = [0] * (numImages * numLine)
    for samples in imageSample['sampleArray']:
        for bbox in samples['BoundingBox']:
            bbArray[i] = bbox
            i = i + 1

# print(data)
#
# def getPixel(index,bbArray,imgePath):
#     image = Image.open(imagePath+index+'.png')
#     pixelMatrixArray = [0] * len(bbArray)
#     for bb in bbArray:
#         for x in

