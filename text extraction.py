import pytesseract
import cv2
import json
import tempfile
import os
from PIL import Image

image = input("Enter the image path: ").strip().strip('"')

if not os.path.exists(image):
    print("Error: File not found. Please check the path and try again.")
    exit()

file = Image.open(image)
try:
    img = cv2.imread(image)

    width = 800
    height = int((width / img.shape[1]) * img.shape[0]) 
    resized_img = cv2.resize(img, (width, height))

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary_inv, kernel, iterations=1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, dilated)

    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(temp_path, config=custom_config).replace("\t", " ").strip()
    os.remove(temp_path)

    text_lines = [line.strip() for line in text.split("\n") if line.strip()]

    json_output = {"Extracted Text": text_lines}

    print("\nJSON Output:")
    print(json.dumps(json_output, indent=4, ensure_ascii=False))

    with open("extracted_text.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    print("\nExtracted text:")
    print(text)
    print("JSON and text files saved successfully.")

except Exception as e:
    print(f"Error during processing: {e}")
