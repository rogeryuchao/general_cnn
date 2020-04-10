from __future__ import absolute_import, division, print_function
import tensorflow as tf
import math
import os
import datetime
import sys

# User defined packages
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_root_dir, log_root_dir,  GLOBAL_LEARNING_RATE, \
    WEIGHT_DECAY, THRESHOLD
from prepare_data import generate_datasets, load_and_preprocess_image
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet
from models.model_selection import get_model


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels

def folder_preparation(job_id, product_id):
    # Genearte log file path and precreate log file header
    log_dir = log_root_dir + job_id + "/" + product_id  + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file = open(log_dir +"training_result_step" + ".log", "w")
    file.write("type\t")
    file.write("timestamp\t")
    file.write("epoch\t")
    file.write("step\t")
    file.write("train_accuracy\t")
    file.write("predict_labels\t")
    file.write("actual_labels\n")
    file.close()
    
    file = open(log_dir +"training_result" + ".log","w")
    file.write("timestamp\t")
    file.write("epoch\t")
    file.write("valid accuracy\n")
    file.close()
            
    # Generate save model path
    save_model_dir = save_model_root_dir + job_id + "/" + product_id + "/"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
        
    return log_dir, save_model_dir

def main(argv):
    # Need the user to provide system argv for job_id and product_id, it is prepared for frontend calling
    if len(argv) < 2 or len(argv) > 3:
        print("ERROR: Format error, refer to the usage: python test.py job_id product_id")
    elif not argv[1].isdigit():
        print("ERROR: Format error, job_id must be in int format")
    elif not argv[1].isalnum():
        print("ERROR: Format error, product_id must be consistent by character or number, without special character")
    else:
        print("INFO: Start training model " + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        # GPU settings
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Folder generate for log file and model saving
        log_dir, save_model_dir = folder_preparation(argv[1], argv[2])                
             
        # get the dataset
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

        # create model
        model = get_model()
        print_model_summary(network=model)
    
        # Setup target for validation dataset accuracy, only when the valid_accuracy reachs the threshold the weight can be saved
        threshold = THRESHOLD

        # define loss calculation
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
        # Tried RMSprop for optimizer, the result is not so good, finetune the optimizer to Adam or Momentum
        optimizer = tf.keras.optimizers.Adam(lr = GLOBAL_LEARNING_RATE, decay = WEIGHT_DECAY)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate = GLOBAL_LEARNING_RATE,momentum = MOMENTUM)

        # Define training KPI
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # Define valid KPI
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

        # @tf.function
        def train(image_batch, label_batch):
            with tf.GradientTape() as tape:
                predictions = model(image_batch, training=True)
                loss = loss_object(y_true=label_batch, y_pred=predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

            train_loss.update_state(values=loss)
            train_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        
            return predictions.numpy(), tf.math.argmax(predictions, axis =1).numpy()

        # @tf.function
        def valid(image_batch, label_batch):
            predictions = model(image_batch, training=True)
            v_loss = loss_object(label_batch, predictions)

            valid_loss.update_state(values=v_loss)
            valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        
            return tf.math.argmax(predictions, axis =1).numpy()

        # start training
        for epoch in range(EPOCHS):
            train_step = 0
            valid_step = 0
            for features in train_dataset:
                train_step += 1
                images, labels = process_features(features, data_augmentation=False)
                predictions, predict_labels = train(images, labels)
                
                # Print the info on the screen for developer to monitor training detail
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}, softmax(logits):{}, "
                      "predict_label:{}, target_label:{}".format(epoch,
                                                                EPOCHS,
                                                                train_step,
                                                                math.ceil(train_count / BATCH_SIZE),
                                                                train_loss.result().numpy(),
                                                                train_accuracy.result().numpy(),
                                                                predictions,
                                                                predict_labels,
                                                                labels))
                
                # Record information into the log file
                file = open(log_dir +"training_result_step" + ".log", "a")
                file.write("train\t")
                file.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "\t")
                file.write(str(epoch) + "\t")
                file.write(str(train_step) + "\t")
                file.write(str(train_accuracy.result().numpy()) + "\t")
                file.write(str(predict_labels) + "\t")
                file.write(str(labels) + "\n")
                file.close()

            for features in valid_dataset:
                valid_step += 1
                valid_images, valid_labels = process_features(features, data_augmentation=False)
                predict_labels = valid(valid_images, valid_labels)
                
                file = open(log_dir +"training_result_step" + ".log", "a")
                file.write("validation\t")
                file.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "\t")
                file.write(str(epoch) + "\t")
                file.write(str(valid_step) + "\t")
                file.write(str(valid_accuracy.result().numpy()) + "\t")
                file.write(str(predict_labels) + "\t")
                file.write(str(labels) + "\n")
                file.close()
                
            # Print the info on the screen for developer to monitor validation result
            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                  "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                      EPOCHS,
                                                                      train_loss.result().numpy(),
                                                                      train_accuracy.result().numpy(),
                                                                      valid_loss.result().numpy(),
                                                                      valid_accuracy.result().numpy()))
            # Create log file in txt format, easy for pandas to analysis and for best model selection
            file = open(log_dir +"training_result" + ".log","a")
            file.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "\t")
            file.write(str(epoch) + "\t")
            file.write(str(valid_accuracy.result().numpy()) + "\n")
            file.close()
            
            valid_accuracy_result = valid_accuracy.result().numpy()
        
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            
            # Save the weights for evaluation and prediction only when the valid accuracy is higher than threshold and best ever result
            if epoch % save_every_n_epoch == 0:
                if valid_accuracy_result >= threshold:
                    model.save_weights(filepath=save_model_dir+str(epoch)+"/model", save_format='tf')
                    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
                    # tf.keras.models.save_model(model, save_model_dir + str(epoch), save_format='tf')
                
                    # Threshold update
                    threshold = valid_accuracy_result
            
            # save the whole model
            # tf.saved_model.save(model, save_model_dir)

            # convert to tensorflow lite format
            # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
            # converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # tflite_model = converter.convert()
            # open("converted_model.tflite", "wb").write(tflite_model)
    

if __name__ == '__main__':
    main(sys.argv)
    

