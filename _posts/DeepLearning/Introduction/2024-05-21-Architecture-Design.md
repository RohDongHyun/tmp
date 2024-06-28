---
title: "Architecture Design"
author: rdh
date: 2024-05-21 05:21:23.532
categories: [Deep Learning, Introduction to Deep Learning]
tags: [deep learning]
math: true
---

## Architecture Design
**Architecture**란 network의 전반적인 구조를 말한다. Neural networks에서의 architecture란 몇 개의 layer를 사용할 것인지, 각 layer 별로 몇 개의 unit을 어떤 식으로 사용할 것인지, 각 unit들을 어떻게 연결할 것인지와 같은 의사결정을 통해 생성되는 network의 구조를 말한다.

![](/assets/img/Architecture-Design-01.png)

## Advantage of Depth
### Layer and Units
일반적인 neural networks는 **layer**라고 불리는 여러 개의 unit의 group들이 chain 구조를 이루는 형태로 구성되어 있다.

이러한 chain-based architecture의 경우 network의 깊이 (layer의 개수)와 각 layer의 넓이 (layer를 이루는 unit의 개수)를 통해 main architecture가 결정된다.

하나의 hidden layer를 갖는 network도 학습이 가능하다. 하지만, network가 더 깊어질수록 학습은 어려워지나 성능은 좋아지는 것이 일반적이다. 

다만, 이러한 architecture의 성능은 이론적인 부분으로 설명하기에는 아직까지 한계가 있어, 일반적으로 이상적인 network architecture는 validation set error를 통해 실험적으로 결정한다.

> 요즘에는 성능이 좋다고 알려져 있는 기존 model들을 가져와서 backbone architecture로 삼는 것이 일반적이다.
{: .prompt-tip}

### Power of Deep Models
일반적으로 network가 깊어질수록 모델의 성능은 증가하게 된다. Layer의 수가 늘어나는 경우, 각 layer가 input의 추상적이고 복잡한 계층적인 특징을 학습할 수 있기 때문이다. 일반적으로 앞 단의 layer들은 low-level 특징을 학습하고, 뒷 단의 layer들은 high-level 특징을 학습한다.

또한, 깊이가 깊어지면 전체 모델의 unit 수를 보다 적게 설정하고도 복잡한 함수를 근사할 수 있다. Shallow 모델에서는 동일한 flexibility를 위해 더 많은 unit이 필요하지만, deep 모델에서는 layer 간의 조합으로 유사한 효과를 낼 수 있기 때문이다. 이렇게 unit 수가 줄어들게 되면 overfitting의 확률 역시 감소하게 된다.

아래는 multi-digit 숫자 인식 문제에 대해 layer 수와 accuracy에 관한 실험적 결과를 나타내는 그래프이다.

![](/assets/img/Architecture-Design-02.png){: width="650"}

아래 그래프 역시 동일한 문제에 대한 실험적 결과이다. Layer 수가 많은 모델이 parameter의 수가 많은 모델보다 더 좋은 성능을 보여주는 것을 알 수 있다.

![](/assets/img/Architecture-Design-03.png){: width="650"}
