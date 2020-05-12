
def get_conv_model():
    model= Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1),
                    padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                    padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                    padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                    padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])
    return model

