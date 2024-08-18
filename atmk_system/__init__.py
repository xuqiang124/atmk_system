import os
import sys
# DIR PATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_PROJECT_ROOT, "formula_embedding")
_MATHBYTE_DIR = os.path.join(_PROJECT_ROOT, "MathByte")
sys.path.append(_LIB_DIR)
sys.path.append(_MATHBYTE_DIR)