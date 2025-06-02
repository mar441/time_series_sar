import tensorflow as tf
import os

def setup_gpu_memory():
    """Configure GPU memory growth and limit."""
    # Disable TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Get physical devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # First, set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Then configure memory limit for the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=17238)]  # 80% of 21548MB
            )
            
            # Print GPU configuration for debugging
            print("GPU Configuration:")
            print(f"Available GPUs: {len(gpus)}")
            for gpu in gpus:
                print(f"GPU: {gpu}")
                print(f"Memory growth enabled: {tf.config.experimental.get_memory_growth(gpu)}")
                print(f"Memory limit: {tf.config.get_logical_device_configuration(gpu)[0].memory_limit / 1024:.2f} MB")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            # Fallback to just memory growth if virtual device configuration fails
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU devices found. Running on CPU.")

def get_model_config():
    """Get optimized model configuration."""
    return {
        'batch_size': 32,  # Further reduced from 64
        'lstm_units': (128, 64),  # Further reduced from (256, 128)
        'dropout_rates': (0.2, 0.2),
        'dense_units': (64,),
        'learning_rate': 0.001,
        'point_embedding_dim': 8,
        'use_point_embeddings': True,
        'use_residual_connections': True
    } 