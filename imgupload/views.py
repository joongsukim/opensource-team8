from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from .detect import *
from django.views.decorators.csrf import csrf_exempt

import time
from absl import app, flags, logging
import cv2
import numpy as np
import tensorflow as tf
from .yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from .yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from .yolov3_tf2.utils import draw_outputs

# Create your views here.
def dog_image_view(request):
    if request.method == 'POST':
        form = ImgForm(request.POST, request.FILES)

        if form.is_valid():
            form.save() # 저장
            return redirect('dog_image/') #/image_upload/dog_image/ 경로로 이동
    else:
        form = ImgForm()
    return render(request, 'dog_image_form.html', {'form': form}) #초기 등록화면. 사진 업로드 버튼

def display_dog_images(request):
    if request.method == 'GET':
        Dog = Photo.objects.all().order_by('-id')[:1] #가장 최근에 저장된 사진
        return render(request, 'display_dog_images.html', {'dog_images': Dog}) #현재 표정 등록된 화면. 분석하기 버튼

def success(request):
    return HttpResponse('successfully uploaded')

@csrf_exempt
def process(request):
    # 이미지 불러와서 감정 분석 처리하기
    # https://heannim-world.tistory.com/39 참고해서 진행
    dog = Photo.objects.all().order_by('-id')[:1]
    #detect(dog, './media/images') #detect('파일 경로', 'output 경로')

    return render(request, "process_result.html")