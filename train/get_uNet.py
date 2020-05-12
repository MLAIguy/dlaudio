def get_UNet(shape=[219, 60, 1]):
    #Instantiate an empty model
    
    Inputs=Input(shape)
    conv1=conv2D(64, 3, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    conv1=conv2D(64, 3, kernel_initializer = 'he_normal', padding = 'same')(conv1)
    
    pool1=MaxPool2D(pool_size=(2,2))(conv1)
    
    conv2=conv2D(128, 3, kernel_initializer = 'he_normal', padding = 'same')(pool1)
    conv2=conv2D(128, 3, kernel_initializer = 'he_normal', padding = 'same')(conv2)
    
    pool2=MaxPool2D(pool_size=(2,2))(conv2)
    
    conv3=conv2D(256, 3, kernel_initializer = 'he_normal', padding = 'same')(pool2)
    conv3=conv2D(256, 3, kernel_initializer = 'he_normal', padding = 'same')(conv3)
    
    pool3=MaxPool2D(pool_size=(2,2))(conv3)
    
    conv4=conv2D(512, 3, kernel_initializer = 'he_normal', padding = 'same')(pool3)
    conv4=conv2D(512, 3, kernel_initializer = 'he_normal', padding = 'same')(conv4)
    
    drop4=Dropout(0.5)(conv4)
    
    pool4=MaxPool2D(pool_size=(2,2))(conv4)
    
    conv5=conv2D(1024, 3, kernel_initializer = 'he_normal', padding = 'same')(pool4)
    conv5=conv2D(1024, 3, kernel_initializer = 'he_normal', padding = 'same')(conv5)
    
    drop5=Dropout(0.5)(conv5)
    
    up6=Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6=concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7=Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7=concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8=Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8=concatenate([conv2,up8], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9=Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9=concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    return model