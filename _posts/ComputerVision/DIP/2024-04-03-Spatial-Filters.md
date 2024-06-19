---
title: Spatial Filters
author: rdh
date: 2024-04-03T08:31:29.581Z
categories: [Computer Vision, Introduction to Digital Image Processing]
tags: [convolution, low-pass filter, high-pass filter, digital image processing, computer vision]
math: true
---

## 2D Convolution
### Linear Space Invariant(LSI) System
![](/assets/img/spatial-filters-01.png){: width="650"}

### 2D Convolution
![](/assets/img/spatial-filters-02.png){: width="650"}

![](/assets/img/spatial-filters-03.png){: width="650"}

정의에 엄밀한 2D Convolution은 image spread function 또는 kernel을 flip해서 elementwise multiplication을 수행해야 한다.

> 하지만, 실제적으로는 kernel을 뒤집지 않더라도 각 pixel을 neighbor와의 liner 조합으로 바꾸는 것에는 차이가 없으므로 kernel을 뒤집어서 곱하지는 않는다. (이미 뒤집혀진 kernel이라고 생각)
{: .prompt-tip}

## Spatial Filters
### Low Pass Filter
#### Smoothing Filter
Box filter 또는 moving average filter라고도 한다.

![](/assets/img/spatial-filters-04.png){: width="650"}

* 장점: noise 제거/감소
* 단점: blur 효과로 인해 detail 손실

> $$\frac{1}{3} \begin{bmatrix} 0 & 0 & 0 \\ 1 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$은 horizontal blur, $$\frac{1}{3} \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$은 vertical blur이다.
{: .prompt-info}

#### Median Filter
![](/assets/img/spatial-filters-09.png){: width="650"}

### High Pass Filter
#### Edge Detecting Filter

![](/assets/img/spatial-filters-05.png){: width="250"}

#### Sobel Filters

![](/assets/img/spatial-filters-08.png){: width="650"}

#### Laplacian Filter

![](/assets/img/spatial-filters-06.png){: width="650"}

![](/assets/img/spatial-filters-07.png){: width="650"}

Image sharpening을 위한 edge enhancing 기법:
  * Original image : $F$
  * Low-pass image : $Fl$
  * High-pass image : $F-Fl = Fh$
	=> Output = $F + k·Fh$
    * $k$: tunable parameter




