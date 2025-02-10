import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter 
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
    
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()
        
        def call(self, input_embedding, validation_embedding):
            # Convert inputs to tensors
            input_embedding = tf.convert_to_tensor(input_embedding)
            validation_embedding = tf.convert_to_tensor(validation_embedding)
            # Return the L1 distance between the embeddings
            return tf.math.abs(input_embedding - validation_embedding)
        
        def compute_output_shape(self, input_shape):
            return input_shape[0]

        def get_config(self):
            base_config = super().get_config()
            return base_config
