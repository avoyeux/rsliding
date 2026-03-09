"""
Directory contains code to do a convolution. Also contains the a class to convert the padding
choices from cv2.filter2D to numpy.pad.
"""

from .padding import Padding, BorderType
from .convolution import Convolution
