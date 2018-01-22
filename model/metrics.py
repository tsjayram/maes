from keras import backend as K


# custom loss
def alt_binary_cross_entropy(y_true, y_pred):
    mask = K.abs(1 - 2 * y_true)  # kills 0.5 entries while preserving 0 and 1
    z = K.binary_crossentropy(y_true, y_pred)
    return K.mean(z * mask, axis=-1)


# custom metric
def alt_binary_accuracy(y_true, y_pred):
    mask = K.abs(1 - 2 * y_true)  # kills 0.5 entries while preserving 0 and 1
    z = 1.0 - K.abs(y_true - K.round(y_pred))
    return K.sum(z * mask) / (K.sum(mask) + K.epsilon())
