import tensorflow as tf
import datetime
import sys
import os

from configuration import save_model_root_dir, production_image_list
from prepare_data import load_and_preprocess_image
from train import get_model


def get_single_picture_prediction(model, picture_dir):
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_dir), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class = tf.math.argmax(prediction, axis=-1).numpy()[0]
    return pred_class

def main(argv, model_root_dir = save_model_root_dir):
    save_model_dir = model_root_dir + argv[1] + "/" + argv[2] + "/"
    # Need the user to provide system argv for job_id and product_id, it is prepared for frontend calling
    if len(argv) < 2 or len(argv) > 3:
        print("ERROR: Format error, refer to the usage: python test.py job_id product_id")
    elif not argv[1].isdigit():
        print("ERROR: Format error, job_id must be in int format")
    elif not argv[1].isalnum():
        print("ERROR: Format error, product_id must be consistent by character or number, without special character")
    elif not os.path.exists(save_model_dir):
        print("ERROR: Job_ID or product ID incorrect")
    else:
        print("INFO: Start evaluating model " + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        # GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
        # load the model
        model = get_model()
        model.load_weights(filepath=save_model_dir+"model")
        
        # Image dict to return for RESTful
        result = {}
    
        for image in production_image_list:
            pred_class = get_single_picture_prediction(model, image)
            print(image + ":" + str(pred_class))
            result[image] = pred_class
        return result
    

if __name__ == '__main__':
    main(sys.argv)
