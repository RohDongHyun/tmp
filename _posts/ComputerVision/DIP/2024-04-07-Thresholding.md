---
title: Thresholding
author: rdh
date: 2024-04-07T13:24:54.572Z
categories: [Computer Vision, Introduction to Digital Image Processing]
tags: [thresholding, digital image processing, computer vision]
math: true
---
## Thresholding
**Thresholding**은 가장 간단한 image segmentation으로, intensity가 특정 threshold $T$보다 큰 pixel은 1, 아닌 pixel은 0으로 변환한다.

$$
g(x,y) = 
\begin{cases} 
1 & \text{if } I(x,y) > T \quad \text{(object)} \\ 
0 & \text{if } I(x,y) \leq T \quad \text{(background)}
\end{cases}
$$

* Global thresholding: image 내 모든 pixel에 대해 같은 threshold 설정
* Adaptive thresholding: Local 구역 또는 pixel 별로 threshold 설정

![](/assets/img/thresholding-01.png){: width="550"}

## Otsu's Algorithm
**Otsu's algorithm**은 class간 variance를 최대화하는 global thresholding 기법이다.

* $P_i$: probability that $I(x,y) = i,$ $i = 0, 1, \ldots, L-1$
* $m_G$: global mean $\sum_{i=0}^{L-1} i P_i$
* $\sigma_G^2$: global variance $\sum_{i=0}^{L-1} (i - m_G)^2 P_i$

만약 임의의 threshold $k$로 image를 2개의 class로 나눈다고 하자.

$$
C_1 = \{(x,y) | I(x,y) \leq k\}, \quad C_2 = \{(x,y) | I(x,y) > k\}
$$

이 때, 각 class에 대해 class-conditional mean은 다음과 같다.

$$
m_1 = \frac{\sum_{i=0}^{k} i p_i}{p_1} \quad m_2 = \frac{\sum_{i=k+1}^{L-1} i p_i}{p_2}
$$

최종 threshold $T$는 $k$를 변화시키면서, class 간 variance를 maximize 하는 $k$로 선택한다.

$$
\sigma_B^2 = \sum_{k=1}^2 P_k (m_k-m_G)^2
$$

> $\sigma_B^2/\sigma_G^2$은 separability에 대한 좋은 measure이다. (높을수록 좋음)
{:.prompt-tip}

![](/assets/img/thresholding-02.png){: width="650"}

> Otsu's algorithm은 strong peak가 없거나, object가 배경에 비해 너무 작으면 잘 동작하지 않을 수 있다.
{:.prompt-info}

> Lowpass filter 적용 후에, Otsu's algorithm을 적용하거나, edge 근처의 pixel들에 대해서만 Otsu's algorithm을 적용하는 것도 방안
{:.prompt-tip}

## Variable or Adaptive Thresholding
Image를 여러 block들로 나누고, 각 block 별로 specified function을 각각 적용하는 방식이다.

단, Block size 선정이 쉽지 않으며, Blocking artifacts 발생 가능성이 있다는 단점이 있다.