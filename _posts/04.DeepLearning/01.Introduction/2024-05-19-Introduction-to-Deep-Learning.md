---
title: "Introduction to Deep Learning"
author: rdh
date: 2024-05-19 08:06:08.661
categories: [04. Deep Learning, 01. Introduction to Deep Learning]
tags: [deep learning]
math: true
---
## Deep Learning as a Machine Learning

**Machine Learning (ML)**은 주어진 x (predictor)와 y (response)에 대해 function f (즉, y=f(x))를 data로부터 배우는 것을 말한다. 이는 전통적인 programming 방식인 주어진 x와 f()로부터 y를 계산하는 것과 다르다.

**Deep Learning (DL)**은 이러한 f (특히 non-linear function에 대해)를 배우는 것에 있어서 굉장히 좋은 성능을 보여준다. 특히, data size가 크고 model이 deep해질수록 (layer 수와 parameter 수가 커질수록) DL이 일반적인 ML보다 좋은 성능을 보여준다.

![](/assets/img/introduction-to-deep-learning-01.png){: width="650"}


일반적으로 ML은 data를 어떻게 표현하는 지에 따라 성능이 크게 좌우된다. 따라서, 대체로 feature engineering이 algorithm보다 더 중요한 역할을 수행한다. 그리고 feature를 정하는 것은 창의성과 굉장한 시간 및 노력을 요하기에 상당히 어려운 일이다.

이에 반해 DL은 model이 data의 feature를 스스로 찾아낸다. 즉, 따로 feature에 대한 고민을 따로 할 필요가 없이, model을 어떻게 설계하는 지에 대해서만 집중하게 된다. 이는 일반적인 ML model에 비해 DL이 갖는 큰 특징이자 장점이다.

일반적인 ML이 다음과 같은 process를 갖는다면 (파란 영역이 auto로 수행되는 영역),

![](/assets/img/introduction-to-deep-learning-02.png){: width="650"}

DL은 다음과 같은 process를 갖는다.

![](/assets/img/introduction-to-deep-learning-03.png){: width="650"}

## Overview of Deep Learning
DL은 brain, 특히 neuron의 동작 방식을 모사한 concept을 가진다. 그렇기에 대부분의 경우 DL model을 **Neural Network (NN)**이라고도 말한다. Neuron 하나의 동작 방식은 매우 간단하나, 이를 유기적으로 결합한 NN은 복잡한 representation을 표현한다.

![](/assets/img/introduction-to-deep-learning-04.png){: width="300"}

DL은 특히 layer의 수와 parameter의 수가 많아질수록 성능이 좋아지는 특징이 있다. Layer 하나가 일반적인 computation 명령의 집합이라고 본다면, Layer가 깊어지고 parameter의 수가 많아질수록 복잡한 computation 집합을 수행할 수 있게 된다. 우리가 마주하는 수많은 어려운 문제들은 복잡한 computation이 필요한 경우가 많기 때문에 DL model의 complexity와 성능은 일반적으로 비례한다고 직관적으로 이해해보고자 한다.

> ML에 대해 공부하면서 model의 높은 complexity 및 flexibility는 필연적으로 overfitting을 유발한다고 하였다. 허나 일반적인 DL model의 경우 매우 많은 양의 data를 학습에 사용하기 때문에 대부분의 경우 extrapolation이 아니라 interpolation이 된다. 하지만, 여전히 data 수가 부족하거나 data의 distribution이 달라지는 경우에는 overfitting은 큰 문제가 된다.
{: .prompt-info}

### Representation Learning
여기서 잠시, DL에 대한 이해를 더 높이기 위해 representation learning이라는 개념을 소개한다. 일반적으로 어떤 operation 또는 task의 난이도는 우리가 가진 information 또는 data를 어떻게 processing하여 표현(represent)하느냐에 따라 결정된다. 앞서 ML의 성능이 feature를 어떻게 결정하는 지에 따라 좌우된다라는 점과 동일하다.

Representation learning이란 이러한 data의 representation을 data로부터 배우는 것을 의미한다. 즉 input data로부터 각 layer에서 나타나는 new representation (또는 new feature)가 얼마나 data를 잘 표현하는지가 학습의 성능을 결정한다. DL은 결국 representation learning의 일종으로 볼 수 있다.

> 초기 DL model을 이해하고 개선하려는 관점에서 representation learning 관점의 분석이 많았다. 허나, 요즘 DL 특히 Large Language Model (LLM)의 등장으로 DL model의 내부를 이해하려는 노력은 많이 줄어든 것 같다.
{: .prompt-info}

![](/assets/img/introduction-to-deep-learning-05.png){: width="650"}
