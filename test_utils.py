from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector




# Compare the two inputs
def comparator(learner, instructor):
    layer = 0
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  f"at layer: {layer}",
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            #print("b",b) #debug
            #print("a",a) #debug
            raise AssertionError("Error in test") 
        layer += 1
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
# def summary(model):
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     result = []
#     for layer in model.layers:
#         print("layer.name",layer.name)
#         print("layer.output",layer.output)
#         descriptors = [layer.__class__.__name__, layer.output.shape, layer.count_params()]
#         if (type(layer) == Conv2D):
#             descriptors.append(layer.padding)
#             descriptors.append(layer.activation.__name__)
#             descriptors.append(layer.kernel_initializer.__class__.__name__)
#         if (type(layer) == MaxPooling2D):
#             descriptors.append(layer.pool_size)
#             descriptors.append(layer.strides)
#             descriptors.append(layer.padding)
#         if (type(layer) == Dropout):
#             descriptors.append(layer.rate)
#         if (type(layer) == ZeroPadding2D):
#             descriptors.append(layer.padding)
#         if (type(layer) == Dense):
#             descriptors.append(layer.activation.__name__)
#         if (type(layer) == LSTM):
#             descriptors.append(layer.input_shape)
#             descriptors.append(layer.activation.__name__)
#         if (type(layer) == RepeatVector):
#             descriptors.append(layer.n)
#         result.append(descriptors)
#     return result

def summary(model):
    # Compile the model to ensure it has a defined structure for summarization
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    result = []
    
    # Iterate over each layer in the model
    for layer in model.layers:
        #print("Layer name:", layer.name) #debug
        #print("Layer output:", layer.output) #debug
        
        # Handle cases where layer.output is a list of tensors
        if isinstance(layer.output, list):
            output_shapes = [output.shape for output in layer.output]
        else:
            output_shapes = layer.output.shape

        # Initialize descriptors with layer class name, output shapes, and parameter count
        descriptors = [layer.__class__.__name__, output_shapes, layer.count_params()]
        
        # Add additional layer-specific properties
        if isinstance(layer, Conv2D):
            descriptors.extend([
                layer.padding,
                layer.activation.__name__,
                layer.kernel_initializer.__class__.__name__
            ])
        
        if isinstance(layer, MaxPooling2D):
            descriptors.extend([
                layer.pool_size,
                layer.strides,
                layer.padding
            ])
        
        if isinstance(layer, Dropout):
            descriptors.append(layer.rate)
        
        if isinstance(layer, ZeroPadding2D):
            descriptors.append(layer.padding)
        
        if isinstance(layer, Dense):
            descriptors.append(layer.activation.__name__)
        
        if isinstance(layer, LSTM):
            descriptors.extend([
                [output.shape for output in layer.output],  # Shapes of each output in the list
                layer.activation.__name__
            ])
        
        if isinstance(layer, RepeatVector):
            descriptors.append(layer.n)
        
        # Append the descriptors to the result list
        result.append(descriptors)
    
    return result
