---
title: Parameter Estimation and Hypothesis Test
author: rdh
date: 2024-02-03 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [parameter estimation, hypothesis test, statistics]
math: true
---
The process of using information from a sample to make inferences or draw conclusions about a population is called **statistical inference**. The two main types of statistical inference are **estimation** and **hypothesis testing**. 

## Parameter Estimation
**Estimation** refers to the process of using a sample to provide an estimate of a population parameter along with an associated error.

* **Population parameter ($\theta$)** : a representative value that characterizes the population

* **Random sample** : a set of independent and identically distributed (i.i.d.) random variables. The actual values obtained from the sample are called **observations**.

Parameter estimation includes **point estimation** and **interval estimation**.

### Point Estimation
Point estimation involves providing a single estimate for a population parameter calculated from the sample. This estimated value is called the **estimator**.

* Estimator for the population mean : the sample mean $\hat{\mu} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$

* Estimator for the population variance : the sample variance $\hat{\sigma}^2 = S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2$

#### Evaluation of Estimator
Several criteria are used to evaluate an estimator:

1. **Bias**
    * $\text{Bias}(\hat{\theta}) = E(\hat{\theta}) - \theta$
    * An estimator with a smaller bias is considered better.
      * **Unbiased estimator**: an estimator satisfying $E(\hat{\theta}) = \theta$.

> Both the sample mean and the sample variance are unbiased estimators of the population mean and the population variance, respectively.
{: .prompt-info}

2. **Standard error (SE)**
    * $\text{SE}(\hat{\theta})$: the standard error of an estimator $\hat{\theta}$.

When considering both bias and SE, the **mean squared error (MSE)** is often used as a comprehensive evaluation metric:

$$
\begin{aligned}
\text{MSE}(\hat{\theta}) &= \text{Var}(\hat{\theta}) + (\text{Bias}(\hat{\theta}))^2 \\
&= (\text{SE}(\hat{\theta}))^2 + (\text{Bias}(\hat{\theta}))^2
\end{aligned}
$$

### Interval Estimation
**Interval estimation** provides an estimate for a population parameter in the form of a range or interval. A common method for interval estimation is the **confidence interval (CI)**.

#### Confidence Interval (CI)
A CI with a **confidence level** of $100(1 − \alpha)$% is represented as $(L, U)$, where:

$$
P(L\leq\theta\leq U) = 1 - \alpha
$$

Here, $L$ and $U$ are determined from the sample, *i.e.*, $L \equiv L(X_1, · · · , X_n)$ and $U \equiv U(X_1, · · · , X_n)$, making $(L, U)$ a **random interval**.

* $1-\alpha$ is called the coverage probability.

* When the population variance $\sigma^2$ is known, the $100(1-\alpha)$% CI for the population mean $\mu$ of a normal population is given by:

$$
\left(\bar{X} - Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \bar{X} + Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}\right)
$$

#### Meaning of CI
The meaning of a $100(1-\alpha)$% CI for $\mu$ is as follows:

Out of 100 CIs constructed from 100 different random samples:

$$
\left(\bar{X} - Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \bar{X} + Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}\right),
$$

approximately $100(1-\alpha)$ intervals are expected to contain the true population mean $\mu$.

> If $\sigma$ is unknown and must be estimated, the length of the CI will vary depending on the sample.
{: .prompt-info}

#### The Number of Samples
The required sample size for estimation can be determined such that the margin of error below a specified level.

If the population variance $\sigma^2$ is known, the margin of error for the CI of the population mean in a normal distribution is $Z_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}$.

To ensure the margin of error is less than or equal to a specified value $d$, the required sample size $n$ is determined as:

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

## Hypothesis Test
**가설 검정 (Hypothesis Test)** 또는 **유의성 검정 (Test of Significance)**은 기존의 이론이나 법칙을 부정하는 것으로 보이는 현상이 관측되었을 때, 이를 유지할지 부정할지를 결정하기 위한 통계적 결정 방법이다.

반증을 찾기 위해 설정된 가설 (주로 '기존의 가설')을 **귀무가설 (Null hypothesis, $H_0$)**이라고 하고, 귀무가설의 대안으로 상정되는 가설을 **대립가설 (Alternative hypothesis, $H_1$)**이라고 한다.

이 때, 귀무가설에 대한 반증의 강도를 제공하는 과정을 가설 검정이라 한다.

> 대립 가설이 일반적으로 우리가 주장하고자 하는 가설이 된다.
{: .prompt-info}

### Definitions of Terms

#### Types of Hypotheses

* 단순가설 (simple hypothesis): $\theta = \theta_0, p = 0.5$와 같이 모수를 특정값으로 가정하는 가설 (주로 $H_0$)

* 복합가설 (composite hypothesis): 모수값이 하나보다 많은경우를 가정하는 가설

* 단측가설 (one-sided): $\theta > \theta_0$, 또는 $\theta < \theta_0$와 같이 비교하는 값의 한 쪽에 대해서만 제시되는 가설 (주로 $H_1$)

* 양측가설 (two-sided): $\theta \neq \theta_0$와 같이 양 쪽에 대해서 제시되는 가설 (주로 $H_1$)

#### Types of Errors
* **1종 오류 (type Ⅰ error)**: 귀무가설이 옳은 상황에서 귀무가설을 기각함으로 인해 생기는 오류

* **2종 오류 (type Ⅱ error)**: 귀무가설이 틀린 상황에서 귀무가설을 기각하지 못함으로 인해 생기는 오류

#### Significance Level and Power
* **유의수준 (significance level)**: 1종 오류가 일어날 확률, $\alpha$
    * $\alpha$ = 0.05라 함은, 귀무가설이 참인데 기각할 오류를 5% 이하로 하겠다는 것

* **검정력 (power)**: 귀무가설이 거짓일때 test가 귀무가설을 기각할 확률
    * 2종 오류가 일어날 확률을 $\beta$라고 하면, 검정력은 $1-\beta$

#### Test Statistics and Critical Region
* **검정통계량 (Test statistics)**: 가설 검정에 사용되는 통계량

* **기각역 (Critical region)**: 귀무가설 $H_0$을 기각시킬 수 있는 검정통계량의 관측값의 영역

#### P-value
* **유의확률 (P-value)**: 검정 통계량의 관측값을 가지고 귀무가설이 기각되게 하는 가장작은 유의수준. 또는 검정통계량의 관측값을 포함하는 기각역의 최소확률
    * 표본으로부터 구한 검정통계량의 관측값으로 구한 유의확률이 지정된 유의수준 이하로 나타나면 통계적으로 유의하다라고 표현

### Procedure of Hypothesis Test
가설 검정에서 알아두어야 할 사항으로는 다음과 같은 것들이 있다.

* 검정통계량이 다르면 다른 test

* 기각역의 형태는 대립가설의 영향을 받음

* P값(유의확률) 또는 기각역을 구하기 위해서는 귀무가설하에서 검정통계량의 분포를 알아야 함

* 오류의 위험이 더 큰 경우를 1종 오류가 되도록 정함. 즉, 일반적으로 기존에 믿어오던(알려져 있던) 사실을 귀무가설로 설정한다.

일반적으로 가설 검정을 진행할 때에는 어떤 테스트를 고를지부터 결정해야 한다. 1종과 2종 오류를 동시에 작게하기는 어렵기 때문에 보통 1종 오류가 일어날 확률을 정하고 (controlling type Ⅰ error), 그중에 2종 오류가 일어날 확률이 적은 test를 고려한다.

> 주어진 유의수준 (1종 오류의 확률) 하에서 검정력이 가장 큰 (2종 오류가 제일 작은) test를 most powerful test라고 부른다.
{: .prompt-info}

가설 검정의 절차는 다음과 같다.

1. 귀무가설, 대립가설, 유의수준을 정한다.

2. 검정방법을 정한 후, 표본을 추출하고 검정통계량의 값을 계산한다.

3. 가설을 기각할 수 있는지 없는지를 판단하고, 결론을 이끌어낸다.
    1. 유의수준으로부터 기각역을 찾아 검정통계량의 값이 기각역에 속하는지 확인하거나,
    2. 검정통계량의 값으로 유의확률을 계산하여 유의수준과 비교한다.

### Test for the Population Mean
모평균 $\mu$에 관한 추론을 할 때는 일반적으로 표본평균 $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$을 사용한다.

1. 모분산이 알려진 경우에는, 정규분포를 이용
    * $H_0$ : $\mu = \mu_0$

    * 검정통계량: $Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \overset{H_0}{\sim} N(0,1)$

    * 검정통계량의 관측값: $z_0$

    * $P(N(0,1) \leq z_{\alpha}) = \alpha$

    |  대립가설 $H_1$  |               유의확률                |    유의수준 $\alpha$의 기각역    |
    | :--------------: | :-----------------------------------: | :------------------------------: |
    |  $\mu > \mu_0$   |             $P(Z > z_0)$              |        $Z > z_{1-\alpha}$        |
    |  $\mu < \mu_0$   |             $P(Z < z_0)$              |       $Z < -z_{1-\alpha}$        |
    | $\mu \neq \mu_0$ | $P(\vert Z \vert > \vert z_0 \vert )$ | $\vert Z \vert > z_{1-\alpha/2}$ |

2. 모분산이 알려져있지 않은 경우에는, t-분포 이용 (데이터가 정규분포를 따를 때)

    * $H_0$ : $\mu = \mu_0$

    * 검정통계량: $T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \overset{H_0}{\sim} t(n-1)$
        * $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$

    * 검정통계량의 관측값: $t_0$

    * $T \sim t(k)$일 때, $P(T \leq t_p(k)) = p$

    |  대립가설 $H_1$  |               유의확률               |      유의수준 $\alpha$의 기각역       |
    | :--------------: | :----------------------------------: | :-----------------------------------: |
    |  $\mu > \mu_0$   |             $P(T > t_0)$             |        $T > t_{1-\alpha}(n-1)$        |
    |  $\mu < \mu_0$   |             $P(T < t_0)$             |       $T < -t_{1-\alpha}(n-1)$        |
    | $\mu \neq \mu_0$ | $P(\vert T \vert > \vert t_0 \vert)$ | $\vert T \vert > t_{1-\alpha/2}(n-1)$ |

### Test for the Population Variance
모집단의 분산 $\sigma^2$의 추론은 일반적으로 표본분산 $S^2 = \frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X})^2$ 을 사용한다.

$X_1, X_2, \ldots , X_n$이 정규분포 $N(\mu, \sigma^2)$에서의 랜덤표본일 때,

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$

> 카이제곱분포를 이용한 모분산의 추론은 정규모집단 가정에 민감하므로 정규성 가정에 대해 충분히 검토한 후 사용한다.
{: .prompt-warning}

* $H_0$ : $\sigma^2 = \sigma^2_0$

* 검정통계량 : $X^2 = \frac{(n-1)S^2}{\sigma^2_0} \overset{H_0}{\sim} \chi^2(n-1)$

* 검정통계량의 관측값: $X^2_0$

* $X^2 \sim \chi^2(k)$일 때, $P(X^2 \leq \chi^2_p(k)) = p$

|       대립가설 $H_1$       |                            유의확률                             |                          유의수준 $\alpha$의 기각역                           |
| :------------------------: | :-------------------------------------------------------------: | :---------------------------------------------------------------------------: |
|  $\sigma^2 > \sigma^2_0$   |                        $P(X^2 > X^2_0)$                         |                        $X^2 > \chi^2_{1-\alpha}(n-1)$                         |
|  $\sigma^2 < \sigma^2_0$   |                        $P(X^2 < X^2_0)$                         |                         $X^2 < \chi^2_{\alpha}(n-1)$                          |
| $\sigma^2 \neq \sigma^2_0$ | $2P(X^2 > X^2_0)$ 또는 <br> $2P(X^2 < X^2_0)$에서 1보다 작은 값 | $$X^2 > \chi^2_{1-\alpha/2}(n-1)$$ 또는 <br> $$X^2 < \chi^2_{\alpha/2}(n-1)$$ |
