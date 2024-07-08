---
title: "Transfer Learning"
author: rdh
date: 2024-05-30 09:07:26.673
categories: [04. Deep Learning, 02. Techniques for DL]
tags: [transfer learning, pretraining, fine-tuning, deep learning]
math: true
---
## Transfer Learning

Deep learning 분야가 크게 성장할 수 있었던 이유 중 하나는 바로 **transfer learning (전이 학습)**이다. 이는 우리가 풀고자하는 새로운 문제를 해결하고자 할 때, 새로운 모델을 만들고 이를 처음부터 학습하는 것이 아니라 기존에 학습된 모델을 활용하는 방식을 말한다. 이 때, 기존에 학습된 모델에서 풀려고 했던 문제를 **upstream task** 또는 source task라고 하며, 새롭게 풀고자하는 문제를 **downstream task** 또는 target task라고 한다.

> 요즘에는 transfer learning을 이용하여 모델을 구성하고 학습을 진행하는 것이 더 일반적이다. 따라서, 풀고자하는 문제가 있는 경우 모델을 새로 구성하기 전에 어떤 식으로 transfer learning이 가능할 지를 확인해보는 것이 더 좋다.
{: .prompt-tip}

Transfer learning은 크게 **pretraining**과 **fine-tuning**의 두 단계로 나눌 수 있다.

## Pretraining
Pretraining은 large scale dataset을 이용하여 model을 미리 학습시키는 단계를 말한다. Pretraining은 일반적으로 보다 간단한 task에 대해서 학습을 진행하게 되며, 모델이 다양한 pattern과 특징을 학습하여 input들에 대한 기본적인 지식을 갖게하는 것이 목적이다.

예를 들어 image 처리에 관한 task를 학습하는 모델을 만든다고 하자. 이러한 작업을 위한 pretraining으로는 classification 용 CNN model을 ImageNet과 같은 대규모 image dataset에 대해서 학습을 진행하여, 다양한 image의 형태와 패턴 등을 인식하는 능력을 갖게하고, 최종 task를 수행하는 부분의 일부 layer를 제거하여 일종의 feature extractor로써의 역할을 수행하도록 만드는 것이 있을 수 있다.

![](/assets/img/Transfer-Learning-01.png){: width="550"}

Pretraining을 이용하면 시간적인 측면에서 큰 이득을 볼 수 있다. 또한 upstream task와 downstream task가 다르다 할지라도, pretrained model은 mass data에 대한 특징을 학습한 만큼 일반적으로 더 좋은 성능을 발휘한다.

### Greedy Supervised Pretraining
Greedy supervised pretraining이란, pretraining을 시키는 방법론 중 하나로 layer를 하나씩 쌓아가면서 greedy하게 training을 시키는 방법이다. 

Greedy supervised pretraining에서는 우선 shallow model을 학습시킨 후, 해당 model의 parameter를 고정시킨 채 새로운 layer를 얹어 새로운 layer에 대한 parameter만 학습시킨다. 이후, 해당 모델들의 parameter를 고정시키고 새로운 layer를 추가하고 학습하는 작업을 반복하는 것이다.

이러한 방식은 model이 upstream task에 대해서 최적의 성능을 갖도록 보장하는 것은 아니나, pretraining이 빠르게 수행된다는 장점이 있다. 또한, 이후 이어질 fine-tuning 과정을 통해 downstream task의 관점에서 최적의 성능을 보여주는 것이 목적이다.

> 따라서, greedy supervised pretraining에서는 fine-tuning 과정이 필수적이다.
{: .prompt-info}

![](/assets/img/Transfer-Learning-02.png){: width="650"}

## Fine-Tuning
Fine-tuning은 pretrained model을 downstream task에 맞게 조정하는 단계이다. 
Fine-tuning은 한 가지 방법이 존재하는 것이 아니라, 새로운 dataset과 이전 dataset이 비슷한 지, 그리고 새로운 dataset의 양 등에 따라 적절한 fine-tuning 방법이 달라진다.

일반적으로 내가 가진 dataset의 특징에 따른 적절한 fine-tuning 방법은 다음과 같다.

|                    |                 dataset의 분포가 유사                  |            dataset의 분포가 다름             |
| :----------------: | :----------------------------------------------------: | :------------------------------------------: |
| **data 양이 적음** | 추가 학습 없이 downstream task 용 <br> layer 하나 추가 |            _You're in trouble.._             |
| **data 양이 많음** |                   부분적 fine-tuning                   | 전체 모델 (또는 많은 layer) <br> fine tuning |

* 전체 모델 fine-tuning: 모델의 모든 layer를 새 dataset으로 다시 학습시키는 방법이다.

* 부분적 fine-tuning: downstream task를 위해 추가된 layer(보통 상위 layer)만을 학습시키는 방법이다.

![](/assets/img/Transfer-Learning-03.png)







