import numpy as np

class BufferPredictor():
        def __init__(self, buffer_size, number_of_classes) -> None:
                # initialize empty ring buffer for classification
                self.buffer_size = buffer_size
                self.prediction_buffer = np.ones((buffer_size, number_of_classes)) # 4 is number of gestures (idle, swipe_left, swipe_right, rotate_right)
                self.buffer_pointer = 0

        def predict(self, AL):
            self.prediction_buffer[self.buffer_pointer] = AL.T
            self.buffer_pointer = (self.buffer_pointer + 1) % self.buffer_size

            pred = np.argmax(np.prod(self.prediction_buffer, axis=0), axis=0)
            return pred