from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os

json_file=open('model.json','r')
loaded_json=json_file.read()
json_file.close()
model=model_from_json(loaded_json)
model.load_weights('model.weights.h5')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print('loaded model from disk')
def classify(img_file):
    
   
    test_img=image.load_img(img_file,target_size=(64,64))
    test_img=image.img_to_array(test_img)
    test_img=np.expand_dims(test_img,axis=0)
    result=model.predict(test_img)
    if result[0][0]>0.5:
        prediction='Thanos'
    else:
        prediction='joker'
    print(prediction,img_file)


    

files=[]
path=r'D:\ML\cv2\deep\image classify\Datasets\images\test'
for r,d,f in os.walk(path):
    for file in f:
        if file.endswith('.jpeg'):
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('/n')
