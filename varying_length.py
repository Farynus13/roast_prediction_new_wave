import tensorflow as tf

# Example sequences
sequences = [tf.constant([[1, 2], [3, 4], [5, 6]]), tf.constant([[7, 8], [9, 10]]), tf.constant([[11, 12]])]

# Define a generator function
def generator():
    for seq in sequences:
        yield seq

# Create a dataset from the generator
dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(None, 2), dtype=tf.int32))

# Bucket the sequences by length
bucketed_dataset = dataset.bucket_by_sequence_length(
    element_length_func=lambda elem: tf.shape(elem)[0],
    bucket_boundaries=[2, 5],
    bucket_batch_sizes=[2, 2, 2],
    padded_shapes=tf.TensorShape([None, 2])
)

# Example model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 2)),
    tf.keras.layers.LSTM(3, return_sequences=True),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(bucketed_dataset, epochs=10)