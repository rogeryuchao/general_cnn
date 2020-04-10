import tensorflow as tf
import sys
import datetime
import os

from configuration import save_model_root_dir, log_root_dir
from prepare_data import generate_datasets
from train import get_model, process_features

def folder_preparation(job_id, product_id):
    # Genearte log file path and precreate log file header
    log_dir = log_root_dir + job_id + "/" + product_id  + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file = open(log_dir +"test_result_step" + ".log", "w")
    file.write("type\t")
    file.write("timestamp\t")
    file.write("batch\t")
    file.write("test_accuracy\t")
    file.write("predict_labels\t")
    file.write("actual_labels\n")
    file.close()
    
    file = open(log_dir +"test_result" + ".log","w")
    file.write("timestamp\t")
    file.write("test accuracy\n")
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
        print("INFO: Start evaluating model " + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        
        
        # GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
        # Folder generate for log file and model saving
        log_dir, save_model_dir = folder_preparation(argv[1], argv[2]) 

        # get the original_dataset
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
        # load the model
        model = get_model()
        model.load_weights(filepath=save_model_dir+"model")
        # model = tf.saved_model.load(save_model_dir)

        # Get the accuracy on the test set
        loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # @tf.function
        def test_step(images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            test_accuracy(labels, predictions)
        
            return tf.math.argmax(predictions, axis =1).numpy()

        batch = 0
        for features in test_dataset:
            batch += 1
            test_images, test_labels = process_features(features, data_augmentation=False)
            predict_labels = test_step(test_images, test_labels)
            print("loss: {:.5f}, test accuracy: {:.5f}, predict_labels:{}, test_labels:{}".format(test_loss.result(),
                                                                                                  test_accuracy.result(),
                                                                                                  predict_labels,
                                                                                                  test_labels)
                                                                                             )
            
            file = open(log_dir +"test_result_step" + ".log", "a")
            file.write("test\t")
            file.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "\t")
            file.write(str(batch) + "\t")
            file.write(str(test_accuracy.result().numpy()) + "\t")
            file.write(str(predict_labels) + "\t")
            file.write(str(test_labels) + "\n")
            file.close()

        print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))
        
        file = open(log_dir +"test_result" + ".log","a")
        file.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "\t")
        file.write(str(test_accuracy.result()) + "\n")
        file.close()
            

if __name__ == '__main__':
    main(sys.argv)

    