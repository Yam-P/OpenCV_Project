import cv2

print(cv2.__version__)


img = cv2.imread('Capture001.png', 1)

cv2.imshow('Test img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
