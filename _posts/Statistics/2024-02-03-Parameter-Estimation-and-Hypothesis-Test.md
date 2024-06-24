---
title: Parameter Estimation and Hypothesis Test
author: rdh
date: 2024-02-03 11:33:00 +0800
categories: [Statistics, Introduction to Statistics]
tags: [parameter estimation, hypothesis test, statistics]
math: true
---

표본으로부터의 정보를 이용하여 모집단에 관한 추측이나 결론을 이끌어내는 과정을 **통계적 추론 (Statistical Inference)**라고 한다. 이러한 통계적 추론의 종류로는 **추정 (Estimation)**과 **가설 검정 (Hypothesis test)** 또는 **유의성 검정 (Significance test)**이 있다.

## Parameter Estimation
Estimation 이란, 표본으로부터 모집단의 특성값(모수)에 대한 추측값과 오차를 제시하는 것을 말한다. 

* **모수(Population parameter, $\theta$)** : 모집단의 특징을 나타내는 대표값

* **랜덤표본** : 서로 독립이고 동일한 확률분포를 따르는 확률변수들을 말하며, 실제로 표본을 추출하여 얻은 값들을 관측값(Observation)이라 한다.

Parameter estimation의 종류로는 **점추정 (Point estimation)**과 **구간추정 (Interval estimation)**이 있다.

### Point Estimation
Point estimation은 표본으로부터 계산한 모수의 추정값을 제시하는 것을 말한다. 이렇게 제시된 추정 값을 **추정량 (Estimator)**라고 한다.

* 모평균의 추정량 : 표본평균 $\hat{\mu} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$

* 모분산의 추정량 : 표본분산 $\hat{\sigma}^2 = S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2$

#### Evaluation of Estimator
추정량(Estimator)을 평가하는 몇 가지 기준이 있다.

1. 편향 (Bias)
    * $\text{Bias}(\hat{\theta}) = E(\hat{\theta}) - \theta$
    * 편향이 작을 수록 좋은 추정량으로 볼 수 있다.
      * 불편추정량 (Unbiased estimator): $E(\hat{\theta}) = \theta$을 만족하는 추정량

> 표본평균과 표본분산은 각각 모평균과 모분산의 불편추정량이다.
{: .prompt-info}

2. 표준오차 (Standard error), 
    * $\text{SE}(\hat{\theta})$: 추정량 $\hat{\theta}$의 표준편차

Bias와 SE를 동시에 고려한 평가 기준으로는 $\text{MSE}(\hat{\theta})$가 있다.

$$
\begin{aligned}
\text{MSE}(\hat{\theta}) &= \text{Var}(\hat{\theta}) + (\text{Bias}(\hat{\theta}))^2 \\
&= (\text{SE}(\hat{\theta}))^2 + (\text{Bias}(\hat{\theta}))^2
\end{aligned}
$$

### Interval Estimation
Interval Estimation은 모수의 추정값을 구간으로 제공하는 것이다. 구간 추정의 일반적인 방법으로는 **신뢰구간 (Confidence Interval, CI)**이 있다.

#### Confidence Interval (CI)
신뢰수준 (Confidence level)이 $100(1 − \alpha)$%인 신뢰구간 $(L, U)$는 다음을 만족한다.

$$
P(L\leq\theta\leq U) = 1 - \alpha
$$

이 때의 L과 U는 표본으로부터 구해진다. 즉, $L \equiv L(X_1, · · · , X_n)$, $U \equiv U(X_1, · · · , X_n)$이다. 따라서, $(L, U)$ 는 확률 변수로 이루어진 구간 (random interval)이다.

* $1-\alpha$는 포함확률(coverage probability)이라고 부른다.

* 모분산 $\sigma^2$를 알 때 정규모집단의 모평균 $\mu$의 신뢰수준 $100(1-\alpha)$% 신뢰구간은 다음과 같다.

$$
\left(\bar{X} - Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \bar{X} + Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}\right)
$$

#### Meaning of CI
$\mu$의 신뢰수준 $100(1-\alpha)$% 신뢰구간의 의미는 다음과 같다:

100번의 표본 추출을 통해 얻어진 100 개의 신뢰구간

$$
\left(\bar{X} - Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \bar{X} + Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}\right)
$$

에서, $100(1-\alpha)$개 정도의 신뢰구간이 모평균을 포함할 것으로 기대한다.

> 만약 $\sigma$도 몰라서 추정하는경우 신뢰구간의 길이도 표본에 따라 달라진다.
{: .prompt-info}


#### The Number of Samples
추정에 필요한 표본의 수는 미리 정한 오차의 한계가 일정 수준 이하가 되도록 만드는 값으로 정할 수 있다.

만약 모분산을 알고 있는 경우, 정규모집단의 모평균 신뢰구간의 오차의 한계는 $Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}$이 된다.

이 때, 오차의 한계를 $d$ 이하로 하는 표본의 수 $n$은 다음과 같다.

$$
Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}} \leq d \\
\Rightarrow \left( Z_{1-\frac{\alpha}{2}} \frac{\sigma}{d} \right)^2 \leq n
$$

### Maximum Likelihood Estimation
**최대 가능도 추정법 (Maximum Likelihood Estimation, MLE)**은 **가능도 (likelihood)**를 최대가 되게 하는 모수 값을 추정량으로
하는 방법이다.

가능도란, 확률변수의 관측값이 주어졌을때 해당 확률의 정도이자 모수를 가지는 확률 분포의 경우 모수값이 가능한 정도를 말한다. 일반적으로, 모수 $\theta$에 대한 가능도 함수는 다음과 같다.

$$
L(\theta) = L(\theta \mid X_1, \cdots, X_n) = f(X_1, \cdots, X_n; \theta)
$$

즉, 가능도란 확률밀도함수에 확률변수의 관측값을 대입한 값이 된다.

이 때, 주어진 관측값에서 모수의 가능도를 최대가 되게 하는 값으로 모수를 추정하는것이 MLE이다.

$$
\hat{\theta}_{MLE} = \arg\max_{\theta \in \Omega} L(\theta \mid X_1, \cdots, X_n)
$$

예를 들어, $Ber(p)$를 따르는 확률변수 $X_1, X_2, X_3$가 독립적으로 다음과 같이 관측되었다고 하자: $X_1 = 1, X_2 = 0, X_3 = 0$. Joint pdf에 관측값을 대입한 값은 다음과 같다.
$$
p_{X_1, X_2, X_3}(1,0,0) = p \times (1-p) \times (1-p)
$$

따라서, 성공확률 $p$에 대한 가능도는 $L(p) = p(1-p)^2$이 되고, $p$에 대한 MLE는 1/3이 된다.

#### Log-likelihood
가능도 함수에 대해 로그변환을 시킬 경우 MLE를 더 쉽게 구할 수도 있다.

$$
\hat{\theta} = \arg\max_{\theta \in \Omega} L(\theta) = \arg\max_{\theta \in \Omega} \ell(\theta)
$$

특히, 독립인 관측값들이 주어졌을때 더 유용하다.

$$
\ell(\theta) = \log(L(\theta)) = \log\left(\prod f(X_i; \theta)\right) = \sum_{i=1}^{n} \log(f(X_i; \theta))
$$

#### Properties of MLE
MLE는 일반적인 상황에서 variance가 가장 작고, convergence speed는 가장 빠른 추정량이다.

* 일치성(Consistency): $\hat{\theta} \rightarrow \theta$ in probability
* 점근적 정규성(Asymptotic Normality):
    $$\sqrt{n}(\hat{\theta} - \theta) \rightarrow \mathcal{N}(0, \mathcal{I}^{-1}(\theta))$$ in distribution
    * 여기서 $$\mathcal{I}(\theta) = \mathbb{E} \left[ \left( \frac{d}{d\theta} \log f(X; \theta) \right)^2 \right]$$는 피셔정보 (Fisher Information)라고 부른다.

> 즉, 점근적으로 (n이 커질때), MLE는 평균이 $\theta$, 분산이 $\frac{1}{\mathcal{I}(\theta)}$인 정규분포를 따른다.
{: .prompt-info}

#### CI with MLE
MLE의 MLE의 점근분포를 이용하여 $\theta$의 신뢰구간을 구할 수 있다.

$\sqrt{n\mathcal{I}(\theta)}(\hat{\theta} - \theta) \sim \mathcal{N}(0, 1)$로부터, 신뢰수준 $100(1-\alpha)$%인 $\theta$의 CI는 다음과 같다.

$$
\left( \hat{\theta} - Z_{1-\alpha/2} \frac{1}{\sqrt{n\mathcal{I}(\hat{\theta})}}, \hat{\theta} + Z_{1-\alpha/2} \frac{1}{\sqrt{n\mathcal{I}(\hat{\theta})}} \right)
$$