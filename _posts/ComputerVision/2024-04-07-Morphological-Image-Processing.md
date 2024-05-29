---
title: Morphological Image Processing
author: rdh
date: 2024-04-07T13:00:45.071Z
categories: [Computer Vision, Digital Image Processing]
tags: [morphological image processing, digital image processing, computer vision]
math: true
---
## Morphological Image Processing
**Morphological image processing**이란, 구조적 요소나 커널을 사용하여 이미지 상의 객체들을 처리함으로써, 이미지에서 형태학적인 특성을 강조하거나 단순화, 추출, 보정하는 작업을 말한다.

`여기서는 Binary image를 가정한다. (e.g., after thresholding)`

### Erosion
**Erosion**: kernel과 정확히 일치하는 경우에 대해서만 1, 아니면 0으로 변환시킴

$$
A \ominus B
$$

![](/assets/img/morphological-image-processing-01.png){: width="650"}

![](/assets/img/morphological-image-processing-02.png){: width="450"}

### Dilation
**Dilation**: erosion의 반대로, kernel과 하나라도 겹치면 1, 아니면 0으로 변환시킴

$$
A \oplus B
$$

![](/assets/img/morphological-image-processing-03.png){: width="650"}

### Opening
**Opening**: erosion 후 dilation
  * Thin structure들을 제거함

$$
A \circ B = (A \ominus B) \oplus B
$$

### Closing
**Closing**: dilation 후 erosion
  * Small hole들을 제거함

$$
A \cdot B = (A \oplus B) \ominus B
$$

![](/assets/img/morphological-image-processing-04.png){: width="650"}

### Boundary Extraction

$$
\partial A = A - (A \ominus B)
$$

![](/assets/img/morphological-image-processing-05.png){: width="450"}

## Morphological Algorithms
### Detection Holes and Corners
![](/assets/img/morphological-image-processing-06.png){: width="650"}

### Hall Filling
![](/assets/img/morphological-image-processing-07.png){: width="650"}

### ETC
![](/assets/img/morphological-image-processing-08.png){: width="650"}
