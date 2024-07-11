---
title: "Transformer and Self-attention"
author: rdh
date: 2024-06-02 02:22:41.674
categories: [04. Deep Learning, 01. Introduction to Deep Learning]
tags: [transformer, self-attention, deep learning]
math: true
---

* 참고
  * [Sequence-to-Sequence and Attention](https://rohdonghyun.github.io/posts/Sequence-to-Sequence-and-Attention/)
  * [Vaswani et al, “Attention is all you need” (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Deep learning, 특히 NLP의 발전은 **self-attention** 기법을 이용한 **transformer** 모델의 등장으로 큰 전환점을 맞이하였다. Transformer는 현재까지도 다양한 LLM의 기반이 되고 있으며, parallel processing을 통한 계산이 용이하다는 점에서 GPU의 발전과도 맞물려 크게 떠오른 모델이다.

## Encoder-decoder in Transformer 
Self-attention에 대한 세부적인 개념을 보기 전, transformer의 구조에 대해서 우선 확인해보자. 기본적으로 encoder-decoder 구조를 따르고 있으며, 여러 encoder를 거쳐 생성된 최종 값들을 decoder의 생성 과정에서 참조한다. 이는 기존의 attention과 유사한 방식으로 볼 수 있다.

![](/assets/img/Transformer-and-Self-attention-01.png)

Encoder와 decoder의 내부 구조는 다음과 같다.

![](/assets/img/Transformer-and-Self-attention-02.png)

Encoder는 input이 들어오면 self-attention layer와 feed forward layer를 거치게 되고, 이러한 encoder가 여러 겹으로 쌓여있을 수 있다. Decoder는 self-attention layer, encoder-decoder attention layer, 그리고 feed forward layer를 갖는다.

우선 각 layer가 어떤 식으로 동작하는 지를 먼저 설명하고, transformer의 세부 구조를 이후에 다시 살펴보고자 한다.

## Self-attention Layer
### Recap: Basic Attention

기존의 attention model은 encoder가 RNN으로 구현되어 있다. 이는 sequence input을 처리하기 위해 선택된 구조이지만, RNN은 구조적으로 sequential하게 계산을 진행해야하기 때문에 parallel processing가 쉽지않고, 그로 인해 일반적으로 모델의 bottleneck이 된다.

![](/assets/img/Transformer-and-Self-attention-03.png){: width="550"}

> RNN 대신 CNN을 사용하는 방법도 생각할 수도 있으나, long-time dependecy를 위해서는 무척 많은 layer가 필요한 점과 서로 다른 neighboring words에 대해서도 같은 weight를 가진다는 점에서 어려움이 있다.
{: .prompt-info}

이러한 관점에서 나타난 것이 self-attention 기법이다. Self-attention은 애초에는 parallel processing을 가능하게 하겠다는 관점에서 등장했으나, 기대 이상의 우수한 성능을 보여주어 등장 이후 현재까지 dominate한 모델이 되었다.

### Self-attention in Encoder
기본적으로 self-attention이란 source sequence가 주어졌을 때, 각 token (또는 word)가 다른 모든 token들과 상호작용하여 중요한 정보에 더 많은 가중치를 부여하는 mechanism을 말한다.

즉, input에 대해서도 attention 기법을 적용하여 단순한 embedding 값이 아닌 query, key, value를 통해 계산한 output을 사용하겠다는 뜻이다. Seq2seq의 attention에서 query는 decoder의 hidden state, key와 value는 encoder의 hidden state였다면, self-attention에서는 query는 input token, key와 value는 source sequence 전체가 된다.

Self-attention은 우선 모든 token에 대해서 embedding을 기반으로 query, key, value를 계산한다. 이 때 사용되어지는 matrix $Q \in R^{n\times d_k}, K \in R^{n\times d_k}, V \in R^{n\times d_v}$는 learnable parameter가 된다.

이후는 attention에서 했던 방식과 동일하게, query와 key를 이용하여 attention score를 구하고, 이를 이용해 attention output ($z_1$)을 산출한다.

![](/assets/img/Transformer-and-Self-attention-04.png)

위 과정을 좀 더 세부적으로 살펴보자. 예를 들어, 만약 'Thinking Machines'라는 글귀가 source sequence로 주어졌을 때, 'Thinking'에 대해 attention output을 계산해보자.

우선 'Thinking'과 'Machines'에 대한 embedding을 통해 query, key, value 값을 각각 계산한다.

이후, 'Thinking'에 대한 query ($q_1$)과 각 token의 key 값 ($k_1, k_2$)에 대한 유사도를 dot product로 계산한다. 

이 때, dot product가 적절한 편차와 크기를 갖도록 key의 차원 수의 제곱근 ($\sqrt{d_k}$)를 이용하여 scaling를 수행한다.

해당 값에 softmax를 적용하여 normalization을 수행하고, 이를 각 token의 value 값 ($v_1, v_2$)과 곱하고, 이를 더하여 최종적인 attention output을 생성한다.

![](/assets/img/Transformer-and-Self-attention-05.png){: width="650"}

Attention과 self-attention은 결국 query, key, value가 달라진 것 말고는 계산이 크게 달라진 점이 없다. 이러한 관점에서 엄청난 혁신은 없다고 볼 수도 있으나, 기본적으로 parallel processing을 가능하게 했다는 점에서 큰 의의를 갖는다.

> 즉, self-attention은 각 input token별로 가장 큰 영향을 끼치는 input token이 어떤 것인지에 대한 정보가 담긴 값을 encoding에 사용한다.
{: .prompt-info}

#### Multi-head Attention
CNN에서의 multi-channel 개념을 self-attention에도 적용할 수 있다. CNN에서 channel의 수를 늘림으로써, 다양한 정보를 뽑아낼 수 있었던 것처럼, self-attention 시에도 복수개의 head를 두어 다양한 정보를 뽑아낼 수 있다. 이러한 기법을 **multi-head attention**이라고 한다. 일반적으로 각 attention head를 거쳐 생성된 output은 concatenate되어 linear layer의 input으로 투입된다.

![](/assets/img/Transformer-and-Self-attention-06.png){: width="450"}

#### Position Embedding
Self-attention은 RNN과 다르게 input token에 대한 순서가 최종 결과에 무관한 구조를 갖는다. 이러한 점은 sequence data를 처리함에 있어서 불합리한 결과를 낳기 때문에, 모델이 position을 인식할 수 있는 장치가 필요하다.

Self-attention에서는 이를 **position embedding**을 통해서 해결하였다. 이는 token의 embedding 과정에서 position에 대한 embedding 값인 positional encoding 값을 더해주는 방법이다.

![](/assets/img/Transformer-and-Self-attention-07.png)

### Self-attention in Decoder
Decoder에서의 self-attention도 encoder에서의 self-attention과 원리는 동일하다. 하지만 구조적으로 decoder에서는 미래 시점의 data를 볼 수가 없기에 이러한 부분을 반영해주어야 한다.

따라서, Decoder에서의 self-attention은 계산적인 효율성을 위해 encoder에서 수행한 self-attention과 같은 식으로 구현을 하되, 미래 시점의 data를 참조하지 못하도록 masking을 한다. 이러한 방식을 **masked self-attention**이라고도 한다. 일반적인 masking 방법은 미래 시점의 data를 모두 -inf로 적용하여 해당 token의 attention output 값을 0으로 만들어버리는 방법이다.

![](/assets/img/Transformer-and-Self-attention-08.png){: width="450"}

### Encoder-Decoder Attention
Decoder에서는 self-attention을 거치고 난 이후, encoder-decoder attention layer를 거치게 된다. 이는 기존의 seq2seq 모델의 attention과 마찬가지로, 각 decoding attention output을 query로, encoding output의 모든 값을 key와 value로 하여 attention을 수행하는 것이다. 특이점은 없으나, 여기서도 $\sqrt{d_k}$를 이용한 scaling을 적용한다.

## Transformer Model
다시 transformer model로 돌아오자. Transformer model을 세부적으로 표현하면 다음과 같다.

![](/assets/img/Transformer-and-Self-attention-09.png){: width="550"}

앞서 설명한 positional encoding, multi-head attention, masked self-attention, encoder-decoder attention을 볼 수 있다.

### Add & Norm
설명하지 않은 부분인 'Add & Norm'은 모델이 안정적이고 더 효율적으로 학습할 수 있도록 도와주는 장치이다. 일반적으로 각 multi-head attention layer 또는 feed forward layer 다음에 위치하여 residual connection (skip connection)과 layer normalization을 수행한다 (참고: [layer normalization](https://rohdonghyun.github.io/posts/Batch-and-Layer-Normalizations/)).