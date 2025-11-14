import cv2
# from paddleocr import PaddleOCR

# ocr = PaddleOCR(lang='en')  # specify language
image = cv2.imread("test_number_plate_image.png")
# result = ocr.ocr(image )

import easyocr
reader = easyocr.Reader(['en'])
text = reader.readtext(image)
print([i[1] for i in text])


# print(result)

# for line in result:
#     print([i[1] for i in line])
#     for word in line:
#         print(word[1][0])   # recognized text


