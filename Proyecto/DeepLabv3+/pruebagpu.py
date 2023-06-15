import tensorflow
from tensorflow.python.client import device_lib
def print_info():
     print('  Versión de TensorFlow: {}'.format(tensorflow.__version__))
     print('  GPU: {}'.format([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']))
     print('  Versión Cuda  -> {}'.format(tensorflow.sysconfig.get_build_info()['cuda_version']))
     print('  Versión Cudnn -> {}\n'.format(tensorflow.sysconfig.get_build_info()['cudnn_version']))

print_info()