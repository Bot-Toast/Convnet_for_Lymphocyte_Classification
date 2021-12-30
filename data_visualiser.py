from matplotlib import pyplot as pyplt

""" 
Metric visualisation code,
Shamelessly poached from Tensorflow's own website: 
https://www.tensorflow.org/tutorials/images/classification#create_a_dataset

"""


def image_plot(image_array):
    fig, axes = pyplt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_array, axes):
        ax.imshow(img)
        ax.axis('off')
    pyplt.tight_layout()
    pyplt.show()


"""
REMOVE RUN_INT FROM FUNCTIONS WHEN NOT LOOP TESTING
"""
def plot_results(epoch_count, model_history):
    acc = (model_history.history['accuracy'])
    val_acc = (model_history.history['val_accuracy'])

    loss = (model_history.history['loss'])
    val_loss = (model_history.history['val_loss'])

    epochs_range = range(epoch_count)

    pyplt.figure(figsize=(8, 8))
    pyplt.subplot(1, 2, 1)
    pyplt.plot(epochs_range, acc, label='Training Accuracy')
    pyplt.plot(epochs_range, val_acc, label='Validation Accuracy')
    pyplt.legend(loc='lower right')
    pyplt.title('Training and Validation Accuracy')

    pyplt.subplot(1, 2, 2)
    pyplt.plot(epochs_range, loss, label='Training Loss')
    pyplt.plot(epochs_range, val_loss, label='Validation Loss')
    pyplt.legend(loc='upper right')
    pyplt.title('Training and Validation Loss')
    pyplt.savefig(f"Model_run_post_norm_test_data" + ".png")
    pyplt.show()
