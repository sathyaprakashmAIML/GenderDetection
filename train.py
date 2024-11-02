from keras.models import Sequential
from keras.layers import Conv2D,Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model=Sequential()
model.add(Input(shape=(64,64,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=90,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True
                                 )

val_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('images/train',
                                               target_size=(64,64),
                                               batch_size=8,
                                               class_mode='binary')

val_set=val_datagen.flow_from_directory('images/val',
                                        target_size=(64,64),
                                        batch_size=8,
                                        class_mode='binary')
model.fit(training_set,
                    steps_per_epoch=8,
                    epochs=20,
                    validation_data=val_set,
                    validation_steps=2)

model_json=model.to_json()
with open('model.json','w')as json_file:
            json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")                                  
                                               
                                               
                                               
                                               

