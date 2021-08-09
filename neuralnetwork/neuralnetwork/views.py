from django.http import request, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import io
from PIL import Image
import numpy as np
from . import neuralModel

model = neuralModel.nn()

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def get_number(request):
    try:
        data = json.loads(request.body)
        # print(data)
    except:
        return JsonResponse({'response':'non ok'},status=400)

    imgdata = base64.b64decode(data['data'].split(",")[-1])
    image = Image.open(io.BytesIO(imgdata))
    image = image.convert('RGB')
    image = image.resize((28,28))
    # image.save("test.png")
    # image.show()
    
    array = []

    for x in range(28):
        temp = []
        for y in range(28):
            pixelRGB = image.getpixel((y,x))
            R,G,B = pixelRGB 
            bightness = ((((R+G+B)/3)/255)-1)

            temp.append(bightness)

            # print(bightness)
        array.append(temp)
    
    np.set_printoptions(suppress=True)
    array = np.asarray(array)

    # reshape to 2d
    mat = np.reshape(array,(28,28))

    # Creates PIL image
    img = Image.fromarray(np.uint8(mat * 255) , 'L')
    img.show()

    array = np.array([array,])



    
    return_data = model.predict(array)

    classes = np.argmax(return_data, axis = 1)
    print()


    return JsonResponse({'response':str(classes)})
