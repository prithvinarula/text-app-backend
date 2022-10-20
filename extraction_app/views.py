"""_summary_

    Returns:
        _type_: _description_
"""

import os
import base64
import cv2
import numpy as np
import pytesseract
from django.http import JsonResponse
from django.shortcuts import render
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from .forms import FileData

# Create your views here.
load_dotenv()

path_to_poppler = os.getenv("PATH_TO_POPPLER")
path_to_tes = os.getenv("PATH_TO_TESSERACT")
print(path_to_tes)

print(os.environ.get("PATH_TO_POPPLER"))
print(os.environ.get("PATH_TO_TESSERACT"))
path_to_poppler = "/opt/homebrew/opt/poppler/bin"
path_to_tes = "/opt/homebrew/bin/tesseract"

a = []


def index(request):
    """_summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """
    form = FileData()
    if request.method == "POST":
        form = FileData(request.POST, request.FILES)
        print(request.FILES)
        if form.is_valid():
            ret = form.save()
            print(ret.file.path)
            a.append(ret.file.path)

            # return HttpResponse("File uploaded successfuly")
    context = {"form": form}
    return render(request, "imageapp/index.html", context)


def ima(request):
    """_summary_

    Returns:
        _type_: _description_
    """
    print(path_to_tes)

    pytesseract.pytesseract.tesseract_cmd = path_to_tes
    print(a)

    pdf_file = a[-1:][0]
    print(pdf_file)
    pages = convert_from_path(
        pdf_file,
        500,
        poppler_path=path_to_poppler,
    )

    res = []

    for page in pages:
        page.save("abc.jpeg")
        image = cv2.imread("abc.jpeg", 1)
        cv2.imwrite("new.jpeg", image)
        extracted = pytesseract.image_to_data(Image.open("new.jpeg"), lang="eng", output_type="data.frame")
        extracted = extracted.replace(r"^\s*$", np.nan, regex=True)
        extracted = extracted.dropna()
        image_data = extracted.to_numpy().tolist()
        res.append({"image_data": image_data})
        print(res)

        # c=1
        # page.save(pdf_file[:-4]+str(c)+".jpg", 'JPEG')
        # c=c+1

    image = open("new.jpeg", "rb")  # open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.b64encode(image_read)
    image_64_encode = image_64_encode.decode("UTF-8")

    data = {"key": "value", "image": image_64_encode, "data_coord": res}
    return JsonResponse(data)


def file_upload(request):
    form = FileData(request.POST, request.FILES)
    print(request.FILES)
    print(path_to_tes)
    if form.is_valid():
        ret = form.save()
        print(ret.file.path)
        a.append(ret.file.path)

    pytesseract.pytesseract.tesseract_cmd = path_to_tes
    print(a)

    pdf_file = a[-1:][0]
    print(pdf_file)
    pages = convert_from_path(
        pdf_file,
        500,
        poppler_path=path_to_poppler,
    )

    response = []

    for page in pages:
        page.save("abc.jpeg")
        image = cv2.imread("abc.jpeg", 1)
        cv2.imwrite("new.jpeg", image)
        extracted = pytesseract.image_to_data(Image.open("new.jpeg"), lang="eng", output_type="data.frame")
        extracted = extracted.replace(r"^\s*$", np.nan, regex=True)
        extracted = extracted.dropna()
        image_data = extracted.to_numpy().tolist()
        response.append({"image_data": image_data})
        # print(response)

        # c=1
        # page.save(pdf_file[:-4]+str(c)+".jpg", 'JPEG')
        # c=c+1

    image = open("new.jpeg", "rb")  # open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.b64encode(image_read)
    image_64_encode = image_64_encode.decode("UTF-8")

    data = {"image": image_64_encode, "data_coord": response}
    print(data)
    return JsonResponse(data)
