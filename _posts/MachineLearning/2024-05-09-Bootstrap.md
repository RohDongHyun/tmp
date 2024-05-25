---
title: Bootstrap
author: rdh
date: 2024-05-09T11:16:36.510Z
categories: [Machine Learning, Introduction to Machine Learning]
tags: [bootstrap, machine learning]
math: true
---
**Bootstrap**은 estimator 또는 learning method의 불확실성을 측정함에 있어서 상당히 유용한 통계적 기법이다. 

사전에 bootstrap을 검색하면 '자기 스스로 하는, 독력(獨力)의'라는 뜻이라고 나온다. 본래 bootstrap이란 부츠 신발에 달려있는 끈을 의미하는데, 18세기 지어진 Rudolph Erich Raspe의 소설 "The Surprising Adventures of Baron Munchausen"에서 남작이 늪에 빠졌을 때 자신의 bootstrap을 잡고 스스로를 들어올리면서 늪을 빠져나왔다는 구절에서 유래했다고 한다. (Newton이 울고갈 일!)

## Bootstrap
만약 우리가 주어진 dataset으로 부터 model에 관한 estimator $\hat{\alpha}$를 알고싶다고 하자.

이 경우, 가장 좋은 방안은 original population으로부터 sample들을 지속적으로 뽑아내어 independent dataset을 늘려가는 것이다. 그러나, 쉽게 생각하더라도 이러한 dataset을 늘리는 것은 현실적으로 쉽지 않다.

**Bootstrap**은 이러한 어려움을 해결해주는 기법이다:

1. original dataset (estimated population)으로부터 sampling with replacement를 반복하여 기존 dataset과 동일한 size의 새로운 dataset인 **bootstrap data sets** ($Z^{\ast i}$)을 생성한다.
2. 각 $Z^{\ast i}$로부터 estimator ($\hat{\alpha}^{\ast i}$)를 계산한다.
3. $\hat{\alpha}^{\ast i}$들로부터 $\hat{\alpha}$의 distribution을 추정한다.

이렇게 계산된 estimator들이 늘어나면 (B개라고 하자), 우리는 $\hat{\alpha}$의 standard error를 다음과 같이 추정할 수 있다.

$$
SE_B(\hat{\alpha}) = \sqrt{\frac{1}{B-1} \sum_{r=1}^B (\hat{\alpha}^\ast _r - \hat{\alpha}^\ast )^2}
$$

![](/assets/img/bootstrap-01.png){: width="650"}

### Pros and Cons of the Bootstrap
* **Bootstrap의 장점**
  * Sample distribution에 대한 가정이 필요 없기 때문에 다양한 종류의 데이터에 적용할 수 있다.

  * 이해와 구현이 쉽다.

  * 복잡한 model의 estimator에 대한 standard error 및 confidence interval(CI)을 구할 때 특히 유용하다.

* **Bootstrap의 한계**
  * Computing resource가 꽤나 필요하다.

  * Sample 크기가 작거나 outlier가 포함되어 있는 경우, bootstrap 결과가 효과적이지 않을 수 있다.
  
  * Time series data에 대해서는 적합하지 않다.
    * 이러한 경우, block bootsrap 기법을 사용한다.

### Train with Bootstrap Datasets

만약 size가 n인 bootstrap dataset을 이용해서 학습을 진행한다고 생각해보자.

Sampling with replacement의 경우, 한 번의 sampling에서 어떤 data가 선택되지 않을 확률은 $(n-1)/n$이다. 이러한 sampling을 n번 수행했을 때, 해당 data가 선택되지 않을 확률은 $(1-\frac{1}{n})^n$이 되며, n이 커질수록 해당 값은 $1/e$ (약 37%)에 수렴한다.

즉, 대략 2/3 정도의 data만이 bootstrap dataset의 sample로 선택된다는 것이다.

이런 overlap이 큰 dataset을 training data로 사용하게 되면, true prediction error가 심각하게 underestimate 될 수 있다.

따라서, bootstrap dataset을 학습에 사용하기 위해서는 weak learner을 조합하여 learning을 진행하는 ensemble 방법이 필수적이다.

> 반대로 말하면, ensemble 방법을 제외하고는 bootstrap dataset을 학습에 사용하지 않도록 주의하자.
{: .prompt-warning}