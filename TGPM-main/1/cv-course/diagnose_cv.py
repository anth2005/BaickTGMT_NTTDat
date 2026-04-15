import sys, traceback
print(sys.executable)
try:
    import cv2
    print("FOUND", cv2.__version__, cv2.__file__)
except Exception:
    print("IMPORT_ERROR")
    traceback.print_exc()
