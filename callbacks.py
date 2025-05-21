from tensorflow import keras

def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            "outputs/models/best_model.h5", 
            save_best_only=True
        )
    ]