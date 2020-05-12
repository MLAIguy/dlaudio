def get_InceptionV3():

    # build the Inception V3 network, use pretrained weights from ImageNet
    # remove top fully connected layers by include_top=False
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)) #original code contains input_shape=(img_width, img_height, 3)

    # build a classifier model to put on top of the convolutional model
    # This consists of a global average pooling layer and a fully connected layer with 256 nodes
    # Then apply dropout and sigmoid activation
    
    
    for layer in base_model.layers:
        layer.trainable = False
        print(layer, layer.name, layer.trainable)

    
    model_top = Sequential()
    model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),  
    #model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(1, activation='sigmoid')) 

    model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

    # Compile model using Adam optimizer with common values and binary cross entropy loss
    # Use low learning rate (lr) for transfer learning
    #plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)


    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    
    

    #opt = optimizers.SGD(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    return model
