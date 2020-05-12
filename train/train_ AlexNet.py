#train with AlexNet

with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
    model = get_InceptionV3()

#checkpoint=ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
#                          save_best_only=True, save_weights_only=False, period=1)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    checkpointer = ModelCheckpoint(filepath='./temp.hdf5', 
                               verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=1000, verbose=1,restore_best_weights = True)
    tensorboard = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=False)


    history = model.fit(X_resized_RGB,y,epochs=300,steps_per_epoch=336,batch_size=32,shuffle=True,
         class_weight=class_weight, validation_data=(X_test_resized_RGB, y_test),
                   callbacks=[checkpointer,earlystopper,tensorboard])

#history = model_all2.fit(X,y,epochs=4000,batch_size=32,shuffle=True,
#         class_weight=class_weight, validation_split=0.1, callbacks=[es, mc])
#model_all.save('firsCNN-2.h5')

    model.save("./model.h5")


fig, ax = plt.subplots()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig('accuracy.pdf')

fig, ax = plt.subplots()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig('loss.pdf')



model=load_model('model.h5')


y_test_predicted = model.predict(X_test_resized_RGB)

fragment_opt_threshold, fragment_AUC = fragment_metrics(y_test, y_test_predicted)


calc_metrics(y_test_predicted, y_test, fragment_opt_threshold)

