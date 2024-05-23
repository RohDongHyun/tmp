---
title: Hyperparameter
author: rdh
date: 2024-05-04T07:48:05.299Z
categories: [Machine Learning, Introduction to Machine Learning]
tags: [hyperparameter, machine learning]
math: true
---
## Hyperparameter
**Hyperparameter**는 model에 대해서 외부(사용자)로부터 직접 설정하는 parameter이다.
  * 일반적으로 model의 dataset으로부터 계산되어지는 parameter는 model parameter라고 한다.

> Hyperparameter는 dataset로부터 최적 값을 계산할 수 없지만, dataset에 따라서 최적의 값이 달라진다.
{: .prompt-info}

### Hyperparameter Optimization
기본적으로 hyperparameter는 empirical하게 최선의 조합을 찾아야 한다. 이 때 주로 두가지 방법으로 hyperparameter optimization이 진행된다.

* **Grid Search**
  * Hyperparameter 값에 대한 후보군을 설정하고, 후보군 사이에서의 최선의 조합을 찾아내기

* **Random Search**
  * Hyperparameter 값에 대한 distribution을 설정하고, random sample을 통해 최선의 조합을 찾아내기

![](/assets/img/hyperparameter-01.png){: width="650"}

> 최선의 조합은 대개 validation set performance를 통해 정해지며, grid 또는 distribution을 계속해서 바꿔보며 best hyperparameter를 찾는다.
{: .prompt-tip}

### Architectural Hyperparameters
위와 같은 model architecture를 결정하는 hyperparameter들을 **architectural hyperparameter**라고 한다.
* Number of hidden layers
* Number of neurons in each hidden layer
* Type of activation functions
* Type and amount of regularization

Architectural hyperparameter의 optimization은 높은 cost로 인해 일반적인 grid 또는 random search 방법을 적용하기가 사실상 불가능하다.

> 따라서, 일반적으로는 previous research에서 제안한 model 구조를 따르는 것을 추천하나, 만약 새로 model을 생성해야 하는 상황이라면 처음엔 simple한 모델이 되도록 설정하고, 이후 모델이 가지는 complexity가 점점 커지도록 바꿔보는 방식을 추천한다.  
> 다만 이때에도 문제와 dataset에 따라 모델의 complexity를 증가시키는 방법이 달라지므로 풀고자하는 문제와 관련된 선행 연구들의 trend를 체크하는 것이 중요하다.
{: .prompt-tip}