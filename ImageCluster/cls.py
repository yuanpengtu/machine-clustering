from tensorflow.keras.layers import Dropout,Dense,Conv2D,GlobalAveragePooling2D,Flatten,Reshape,Activation
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16,VGG19,ResNet50,InceptionResNetV2,DenseNet121,MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import SGD,Adam
import os
class classifier(object):
    def __init__(self,img_dir,split_radio,batch_size,val_batch_size,base_model='vgg16',img_shape=(None,None,3),cls_num=21):
        self.img_shape=img_shape
        self.img_dir=img_dir
        self.base_model=base_model
        self.split_radio=split_radio
        self.batch_size=batch_size if self.img_shape!=(None,None,3) else 1
        self.val_batch_size=val_batch_size if self.img_shape!=(None,None,3) else 1
        self.cls_num=cls_num
    def model(self,freeze=True):
        if self.base_model.lower()=='vgg16':
            base_model=VGG16(weights='imagenet',include_top=False,input_shape=self.img_shape)
            input_layer=base_model.input
            layer=base_model.get_layer('block5_pool').output
        elif self.base_model.lower()=='vgg19':
            base_model=VGG19(weights='imagenet',include_top=False,input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        elif self.base_model.lower()=='resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        elif 'inception' in self.base_model.lower():
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        elif self.base_model.lower()=='mobilenet':
            base_model=MobileNet(weights='imagenet', include_top=False, input_shape=self.img_shape,pooling='avg')
            input_layer=base_model.input
            layer=base_model.output
        else:
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        if self.base_model.lower()=='vgg16':
            layer = Flatten()(layer)
            layer = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(layer)
            layer = Dropout(0.5)(layer)
            layer = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(layer)
            layer = Dropout(0.5)(layer)
            prediction = Dense(self.cls_num, activation='softmax', name='predictions')(layer)
        elif self.base_model.lower()=='mobilenet':
            # layer=GlobalAveragePooling2D()(layer)
            layer=Reshape((1,1,1024),name='reshape_1')(layer)
            layer=Dropout(1e-3)(layer)
            layer=Conv2D(self.cls_num, (1, 1),
                              padding='same',
                              name='conv_preds')(layer)
            layer=Activation('softmax')(layer)
            prediction=Reshape((self.cls_num,))(layer)
        elif self.base_model.lower()=='resnet50':
            layer=GlobalAveragePooling2D()(layer)
            prediction = Dense(self.cls_num, activation='softmax', name='predictions')(layer)
        else:
            layer = GlobalAveragePooling2D()(layer)
            prediction = Dense(self.cls_num, activation='softmax', name='predictions')(layer)
        if freeze==True:
            for layer in base_model.layers:
                layer.trainable=False
        else:
            for layer in base_model.layers:
                layer.trainable=True

        model_finetune = Model(input_layer, prediction)

        return model_finetune
    def train(self,initial_epoch=0,epochs=20,opt='sgd'):
        gen=ImageDataGenerator(
            rotation_range=360,horizontal_flip=True,vertical_flip=True,shear_range=0.2,zoom_range=0.2,
            rescale=1/255,validation_split=self.split_radio
        )
        train_data = gen.flow_from_directory(
            directory=self.img_dir,
            batch_size=self.batch_size,
            target_size=(self.img_shape[0],self.img_shape[1]),
            shuffle=True,
            class_mode='categorical',
            subset='training'
        )
        val_data = gen.flow_from_directory(
            directory=self.img_dir,
            batch_size=self.val_batch_size,
            target_size=(self.img_shape[0],self.img_shape[1]),
            shuffle=True,
            class_mode='categorical',
            subset='validation'
        )
        model_finetune=self.model(freeze=False)
        optimizer=SGD(lr=1e-3,momentum=0.9,nesterov=True) if opt=='sgd' else Adam(lr=0.0001)
        model_finetune.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        model_finetune.summary()
        tb=TensorBoard(log_dir='logs',batch_size=self.batch_size)
        es=EarlyStopping(monitor='val_acc',patience=15,verbose=1,min_delta=0.0001)
        cp=ModelCheckpoint(filepath=os.path.join('model','freeze_vgg16_cls--epoch_{epoch:02d}--val_loss_{val_loss:.5f}--val_acc_{val_acc:.5f}--train_loss_{loss:.5f}--train_acc_{acc:.5f}.hdf5'),
                           monitor='val_acc', save_best_only=True,save_weights_only=True,
                           verbose=1,  period=1)
        lr=ReduceLROnPlateau(monitor='val_acc',patience=5,verbose=1)
        his=model_finetune.fit_generator(
            generator=train_data,
            steps_per_epoch=train_data.samples//self.batch_size,
            validation_data=val_data,
            validation_steps=val_data.samples//self.val_batch_size,
            initial_epoch=initial_epoch,
            epochs=epochs,
            callbacks=[tb,es,cp,lr]
        )
        print(his)

if __name__=='__main__':
    model=classifier(
        img_dir='train_data_2',
        split_radio=0.1,
        batch_size=19,
        val_batch_size=19,
        base_model='vgg16',
        img_shape=(224,224,3),
        cls_num=77
    )
    model.train(initial_epoch=0,epochs=100,opt='sgd')