---
title: Nonparametric Statistics
author: rdh
date: 2024-02-04 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [nonparametric statistics, robust methods, statistics]
math: true
---
In parameter estimation, population distribution (e.g., normal distribution) are typically assumed, and the parameters of the distribution are estimated and tested ([reference](https://rohdonghyun.github.io/posts/Parameter-Estimation-and-Hypothesis-Test/)). However, when it is unreasonable to assume a specific distribution for the population, **nonparametric** methods can be considered to relax distributional assumptions and reduce the likelihood of errors.

## Nonparametric Hypothesis Test
Nonparametric methods relax assumptions about the population distribution as much as possible, typically assuming only continuity and, in some cases, symmetry.

In nonparametric inference, commonly used values are scores based on the **sign** or **rank** of the observations. Instead of using the raw observations themselves, these methods rely on values that are independent of the population distribution.

### One Sample Sign Test
The **sign test** is used to test the location parameter $\mu$ under the following assumptions: 

* Assumptions:
  1. Model: $X_i = \mu + e_i$ ($i=1,\dots, n$), where $\mu$ is the unknown location parameter, and $e$ is the error term.
  2. The $n$ error terms $e$ are i.i.d.
  3. The error term $e$ follows a symmetric distribution about 0.

* Hypotheses:  
  Tests for the location parameter $\mu$ are generally conducted under one of the following hypotheses:
  1. $H_0: \mu = \mu_0$ vs. $H_1: \mu > \mu_0$
  2. $H_0: \mu = \mu_0$ vs. $H_1: \mu < \mu_0$
  3. $H_0: \mu = \mu_0$ vs. $H_1: \mu \ne \mu_0$

The sign test is a simple and oldest nonparametric test for the location parameter. Under the null hypothesis $H_0$, the test relies only on the count of observations greater than $\mu_0$.

The sign test statistic is defined as:

$$
B = \sum_{i=1}^n I(X_i-\mu_0),
$$

where $I(x)$ is the indicator function: 1 if $x>0$, and 0 otherwise. Thus, $B$ represents the number of observations greater than $\mu_0$.

The sign test is conducted as follows:

1. For $H_1: \mu > \mu_0$, reject $H_0$ if $B \geq b(\alpha, n)$.
2. For $H_1: \mu < \mu_0$, reject $H_0$ if $B < b(1-\alpha, n)$.
3. For $H_1: \mu \ne \mu_0$, reject $H_0$ if $B \geq b(\alpha/2, n)$ or $B < b(1-\alpha/2, n)$.

Here, $b(\alpha, n)$ is the upper $100\alpha$-th percentile of the sign test statistic $B$ under $H_0$ for a sample of size $n$, satisfying $P_0[B\geq b(\alpha,n)] = \alpha$.

#### Estimation based on Sign Test
Estimation of the location parameter $\mu$ can be conducted using the sign test through the following steps:

1. Arrange the observations in ascending order:

    $$
    X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}
    $$

2. The sign test-based estimate $\hat{\mu}$ is set as the median of the observations:

    $$
    \hat{\mu} = \begin{cases} 
    X_{(k+1)} & n = 2k + 1, \\ 
    \frac{X_{(k)} + X_{(k+1)}}{2} & n = 2k.
    \end{cases}
    $$

This estimate $\hat{\mu}$ is known as the **Hodges-Lehmann one-sample estimator** for $\mu$, based on the Wilcoxon sign test.

> In general, the power of a test is proportional to the accuracy of the corresponding estimation. However, the power of the sign test is relatively low, which means that the median estimated using the sign test is often imprecise.
{: .prompt-info}

### Wilcoxon Signed-Rank Test
The **Wilcoxon signed-rank test** is one of the most widely used nonparametric methods for testing the location parameter in a one-sample scenario. Unlike the sign test, which only considers whether the observations are greater or less than $\mu_0$, this test also takes the relative magnitude of the observations into account.

Unlike the sign test, which does not require the assumption of symmetry, the Wilcoxon signed-rank test requires the assumption of symmetry in the distribution.


The Wilcoxon signed-rank test is conducted as follows:

1. For all $i = 1, \ldots, n$, calculate $Z_i = X_i - \mu_0$.
2. From $\vert Z_1\vert, \vert Z_2\vert, \ldots, \vert Z_n\vert$, assign ranks $R_i^+$ based on the magnitude of $\vert Z_i\vert$.
3. Compute the Wilcoxon signed-rank test statistic:
    
    $$
    W^+ = \sum_{i=1}^n I(X_i - \mu_0) \cdot R_i^+.
    $$

4. Conduct the test at significance level $\alpha$:
    * For $H_1: \mu > \mu_0$, reject $H_0$ if $W^+ \geq w^+(\alpha, n)$.
    * For $H_1: \mu < \mu_0$, reject $H_0$ if $W^+ \leq w^+(1 - \alpha, n)$.
    * For $H_1: \mu \neq \mu_0$, reject $H_0$ if $W^+ \geq w^+(\alpha/2, n)$ or $W^+ < w^+(1 - \alpha/2, n)$.

Here, $w^+(\alpha, n)$ is the upper $100\alpha$-th percentile of the Wilcoxon signed-rank test statistic $W^+$ under $H_0$.

> In practice, the primary estimators used for location parameters are based on the $t$-test or the Wilcoxon signed-rank test.
{: .prompt-tip}

#### Estimation based on Signed-Rank Test
When the distribution of error terms $e$ is assumed to be symmetric, the point estimation for the location parameter $\mu$ can be derived using the Wilcoxon signed-rank test statistic as follows:

1. For all $i \leq j (i, j = 1, 2, \ldots, n)$, compute:
 
    $$
    W_{ij} = \frac{X_i + X_j}{2},
    $$

    where $W_{ij}$ is called the **Walsh average**.

2. Arrange the Walsh averages $W_{ij}$ in ascending order as order statistics:
    
    $$
    W_{(1)}, W_{(2)}, \ldots, W_{(N)}
    $$
    
    where $N = n(n+1)/2$. Define the point estimation $\hat{\mu}$ for $\mu$ as the median of the Walsh averages:

    $$
    \hat{\mu} = 
    \begin{cases}
    W_{(k+1)} & \text{if } N = 2k + 1, \\
    \frac{W_{(k)} + W_{(k+1)}}{2} & \text{if } N = 2k.
    \end{cases}
    $$

This estimator $\hat{\mu}$ is known as the **Hodges-Lehmann one-sample estimator** for $\mu$, based on the Wilcoxon signed-rank test.

> Due to the computational effort required to calculate Walsh averages, this method is rarely used in practice.
{: .prompt-warning}

### Two-Sample Wilcoxon Rank Sum Test
The problem of estimating and testing the location parameters from two populations based on random samples is referred to as a **two-sample location problem**.

Two independent random samples are drawn, often representing a **control** population and a **treatment** population. The goal is to determine whether there is a difference in the location parameters between two populations and, if so, quantify that difference.

Let the random samples from two populations be $(X_1, X_2, \ldots, X_m)$ and $(Y_1, Y_2, \ldots, Y_n)$, where $m$ and $n$ are the respective sample sizes. Denote $N = m + n$, and assume that $m \geq n$ for convenience.

* Assumptions:
    1. $\mu$ is the unknown location parameter for $X_i$, and $\Delta$ is the shift parameter (difference in location parameters between the two groups), with $e$ as the error term:

    $$
    \begin{aligned}
    X_i &= \mu + e_i, \quad i = 1, \ldots, m \\
    Y_j &= \mu + \Delta + e_{m+j}, \quad j = 1, \ldots, n
    \end{aligned}
    $$

    2. The $N$ error terms $e$ are independent and identically distributed (i.i.d.) and follow a continuous distribution.

* Hypotheses:  
    The estimator for $\Delta$ is denoted as $\hat{\Delta}$. Tests for $\Delta$ typically involve one of the following hypotheses:
    1. $H_0: \Delta = 0$ vs. $H_1: \Delta > 0$
    2. $H_0: \Delta = 0$ vs. $H_1: \Delta < 0$
    3. $H_0: \Delta = 0$ vs. $H_1: \Delta \neq 0$

The **Wilcoxon rank sum test** is the most widely used method for the two-sample location problem. It uses the ranks of the observations in the combined sample, and the procedure is as follows:

1. Assign ranks $R_j$ to the $Y_j$ observations in the combined sample $X \cup Y$, where $X = (X_1, \ldots, X_m)$ and $Y = (Y_1, \ldots, Y_n)$.
2. Compute the Wilcoxon rank sum statistic:
    
    $$
    W = \sum_{j=1}^n R_j
    $$

3. Conduct the test at significance level $\alpha$:
    * For $H_1: \Delta > 0$, reject $H_0$ if $W \geq w(\alpha, m, n)$.
    * For $H_1: \Delta < 0$, reject $H_0$ if $W < w(1 - \alpha, m, n)$.
    * For $H_1: \Delta \neq 0$, reject $H_0$ if $W \geq w(\alpha/2, m, n)$ or $W < w(1 - \alpha/2, m, n)$.
    
Here, $w(\alpha, m, n)$ denotes the $100\alpha$-th percentile of the distribution of the rank sum statistic $W$ under $H_0$.

#### Estimation based on Rank Sum Test
The shift parameter $\Delta$ can be estimated using the Wilcoxon rank sum test statistic as follows:

1. Compute $V_{ij} = Y_j - X_i$ for all $i, j (i = 1, \ldots, m, j = 1, \ldots, n)$.
2. Arrange the $V_{ij}$ values in ascending order to obtain the order statistics $V_{(1)}, V_{(2)}, \ldots, V_{(mn)}$. Define the point estimation of $\Delta$ as the median of the $V_{ij}$ values.

$$
\Delta = 
\begin{cases} 
V_{(k+1)}, & \text{if } mn = 2k + 1, \\
\frac{V_{(k)} + V_{(k+1)}}{2}, & \text{if } mn = 2k.
\end{cases}
$$

### Two-Sample Test for Scale Parameters
The **two-sample scale problem** involves testing whether the scale parameters of two samples differ.

Random samples of sizes $m$ and $n$ from two populations are denoted as $(X_1, X_2, \ldots, X_m)$ and $(Y_1, Y_2, \ldots, Y_n)$. Let $N = m + n$, and assume $m \geq n$ for convenience.

* Assumptions:
    1. $\mu_X$ and $\mu_Y$ are the location parameters, and $\sigma_X$ and $\sigma_Y$ are the scale parameters of the $X$ and $Y$ samples, respectively, with $e$ as the error term:     
    $$
    \begin{aligned}
    X_i &= \mu_X + \sigma_X e_i, \quad i = 1, \ldots, m \\
    Y_j &= \mu_Y + \sigma_Y e_{m+j}, \quad j = 1, \ldots, n
    \end{aligned}
    $$

    2. The $N$ error terms $e$ are i.i.d. and follow a continuous distribution.

* Hypotheses:
    The ratio of the two scale parameters is denoted as $\gamma = \sigma_Y / \sigma_X$. Testing $\gamma$ is generally performed under one of the following hypotheses:

    1. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 > 1$
    2. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 < 1$
    3. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 \ne 1$

In cases where $\mu_Y - \mu_X$ is known, $Y_j$ can be adjusted by subtracting $\mu_Y - \mu_X$. This adjustment ensures that the location parameters of the two populations are the same, allowing the scale parameters to solely influence the ranks in the combined sample.

#### Ansari-Bradley Test
The **Ansari-Bradley test** is a widely used method for the two-sample scale problem, based on the ranks in the combined sample.

1. Combine the $N = m + n$ observations and arrange them in ascending order.

2. For each $X_i$, assign a rank $S_i$ corresponding to its position in the combined sample. The **Ansari-Bradley score** $a_{AB}(S_i)$ is defined as follows:

    | $S_i$         | 1   | 2   | ... | $\frac{N+1}{2}$ | ... | $N-1$ | $N$ (odd) |
    | ------------- | --- | --- | --- | --------------- | --- | ----- | --------- |
    | $a_{AB}(S_i)$ | 1   | 2   | ... | $\frac{N+1}{2}$ | ... | 2     | 1         |

    | $S_i$         | 1   | 2   | ... | $\frac{N}{2}$ | $\frac{N}{2} + 1$ | ... | $N-1$ | $N$ (even) |
    | ------------- | --- | --- | --- | ------------- | ----------------- | --- | ----- | ---------- |
    | $a_{AB}(S_i)$ | 1   | 2   | ... | $\frac{N}{2}$ | $\frac{N}{2}$     | ... | 2     | 1          |

3. Compute the Ansari-Bradley test statistic: 
   
   $$
   T_{AB} = \sum_{i=1}^m a_{AB}(S_i)
   $$

4. Decision rules at significance level $\alpha$ $(\gamma^2 = \sigma_Y^2/\sigma_X^2)$:
    * For $H_1 : \gamma^2 > 1$, reject $H_0$ if $T_{AB} \geq t_{AB}(\alpha, m, n)$.
    * For $H_1 : \gamma^2 < 1$, reject $H_0$ if $T_{AB} < t_{AB}(1 - \alpha, m, n)$.
    * For $H_1 : \gamma^2 \neq 1$, reject $H_0$ if $T_{AB} \geq t_{AB}(\alpha/2, m, n)$ or $T_{AB} < t_{AB}(1 - \alpha/2, m, n)$.

Here, $t_{AB}(\alpha, m, n)$ represents the $100\alpha$-th percentile of $T_{AB}$ under $H_0$.

$\gamma^2 > 1$ $(\sigma_Y^2 > \sigma_X^2)$ means that the $Y$ observations are more dispersed than the $X$ observations. This results in the $X_i$ ranks $S_i$ being concentrated near the center, leading to larger $a_{AB}(S_i)$ scores and a higher $T_{AB}$.

Although $T_{AB}$ is defined as the sum of the Ansari-Bradley scores for $X$, a similar test can be performed by considering the scores for $Y$.

## Correlation Analysis
**상관분석 (Correlation Analysis)**은 두 변수 간에 어떤 선형적 관계를 가지고 있는 지를 분석하는 방법을 말한다. 

### Correlation Coefficient

일반적으로 상관관계의 정도를 나타내는 값으로 다음과 같이 정의된 **상관계수 (Correlation Coefficient) $\rho$**를 사용한다.

$$
\rho = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}
$$

$-1 \leq \rho \leq 1$이고, $\rho$가 1에 가까울수록 강한 양의 상관관계, −1에 가까울수록 강한 음의 상관관계가 있음을 나타낸다. 그리고 X와 Y가 독립인 경우 $\rho=0$이 된다.

이러한 상관계수 $\rho$의 추정량으로는 주로 **피어슨 표본상관계수 (Pearson Correlation Coefficient)**가 사용되며, 이는 다음과 같이 정의된다.

$$
r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \cdot \sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$

다만, $\rho$는 정규분포에서는 상관관계를 나타내는 좋은 측도이지만, 비모수적 방법에서는 $\rho$의 의미가 약해지며 앞의 $\rho$ 성질과 비슷한 성질을 갖는 다른 측도를 사용한다.

그 중 보편적으로 사용되는 측도로는 **켄달의 타우 (Kendall's Tau)**, **스피어만의 순위상관계수 (Spearman's Rank Correlation Coefficient)**가 있다.

### Kendall's Tau
관측값이 $(X_1, Y_1), \ldots, (X_n, Y_n)$으로 주어져 있을 때, $i, j$ 번째 쌍에 대해 다음과 같이 정의하자.

* $X_i - X_j$와 $Y_i - Y_j$의 부호가 같다면 **부합(concordant)**이고, 이에 대한 확률은 $\pi_c = \text{P}[(X_i - X_j)(Y_i - Y_j) > 0]$

* $X_i - X_j$와 $Y_i - Y_j$의 부호가 다르면 **비부합(discordant)**이고, 이에 대한 확률은 $\pi_d = \text{P}[(X_i - X_j)(Y_i - Y_j) < 0]$

이 때 켄달의 타우(Kendall's tau)는 다음과 같이 정의된다.

$$
\tau = \pi_c - \pi_d
$$

즉, concordant 확률과 discordant 확률의 차이를 말한다.

#### Properties of Kendall's Tau

* $\pi_c + \pi_d = 1$이므로, $-1 \leq \tau \leq 1$을 만족

* X, Y가 서로 독립인 경우 $\pi_c=\pi_d=1/2$ 이기 때문에 $\tau = 0$이다. 또한 $\tau > 0$은 X와 Y가 양의 상관관계에 있음을 뜻하고, $\tau < 0$은 X와 Y가 음의 상관관계에 있음을 뜻한다.

* 켄달의 $\tau$와 상관계수 $\rho$사이에는 (X, Y)가 이변량 정규분포를 따를 때, 다음과 같은 관계가 성립한다.

$$
\tau = \frac{2}{\pi} \arcsin (\rho)
$$

#### Kendall's Tau Independence Test
앞의 성질을 이용하여, X와 Y의 독립성을 검정하기 위해 귀무가설 $H_0 : \tau = 0$을 검정한다. 이에 대한 대립가설은 한쪽검정, 양측검정 모두 가능하다.

1. 모든 $i, j$에 대해 부합인 쌍의 개수 $P$와 비부합인 쌍의 개수 $Q$를 계산한다.

    $$
    \begin{aligned}
    P &= (X_i - X_j)(Y_i - Y_j) > 0 \text{인 쌍의 개수} \\
    Q &= (X_i - X_j)(Y_i - Y_j) < 0 \text{인 쌍의 개수}
    \end{aligned}
    $$

2. 켄달 통계량: $K = P - Q = \sum_{i<j} \text{sign}(X_i - X_j) \text{sign}(Y_i - Y_j)$.   
여기서 $t = 0$이면 $\text{sign}(t) = 0$, $t < 0$이면 $\text{sign}(t) = -1$, $t > 0$이면 $\text{sign}(t) = 1$이다.

3. 검정법: 유의수준 $\alpha$에서,
    * $H_1 : \tau > 0$일 때, $K \geq k(\alpha, n)$이면 $H_0$를 기각
    * $H_1 : \tau < 0$일 때, $K \leq -k(\alpha, n)$이면 $H_0$를 기각
    * $H_1 : \tau \neq 0$일 때, $K \geq k(\alpha/2, n)$ 또는 $K \leq -k(\alpha/2, n)$이면 $H_0$를 기각

여기서 $k(\alpha, n)$은 $P_0[K \geq k(\alpha, n)] = \alpha$를 만족하는 상수이다.

#### Estimation of Kendall's Tau
켄달의 타우를 추정하기 위해 다음과 같은 켄달의 표본상관계수를 사용한다.

$$
\hat{\tau} = \frac{K}{\binom{n}{2}} = \frac{2K}{n(n-1)}
$$

### Spearman's Rank Correlation Coefficient
$X_i$와 $Y_i$의 순위를 각각 $R_i$와 $S_i$라고 하자. 스피어만의 순위상관계수 (Spearman's Rank Correlation Coefficient)는 다음과 같이 정의된다.

$$
r_s = \frac{\sum_{i=1}^{n}(R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^{n}(R_i - \bar{R})^2 \cdot \sum_{i=1}^{n}(S_i - \bar{S})^2}}
$$

> 스피어만 순위상관계수는 Rank를 값으로 갖는 피어슨 상관계수이다.
{: .prompt-info}

#### Spearman Independence Test

1. $X_i, Y_i$의 순위 $R_i, S_i$를 앞장과 같이 구한다.

2. 스피어만의 순위상관계수:

    $$
    r_s = 1 - \frac{6}{n(n^2 - 1)} \sum_{i=1}^{n} (R_i - S_i)^2
    $$

3. 검정법: 이미 계산된 $r_s(\alpha, n)$의 값에 대해,
    * $H_1 : \rho_s > 0$일 때, $r_s \geq r_s(\alpha, n)$이면 $H_0$를 기각
    * $H_1 : \rho_s < 0$일 때, $r_s \leq -r_s(\alpha, n)$이면 $H_0$를 기각
    * $H_1 : \rho_s \neq 0$일 때, $r_s \geq r_s(\alpha/2, n)$ 또는 $r_s \leq -r_s(\alpha/2, n)$이면 $H_0$를 기각

### Comparison with Kendall's Tau and Spearman's Rank Correlation Coefficient
* 계산과정은 $r_s$의 계산이 $\hat{\tau}$의 계산보다 간편하다.

* $\hat{\tau}$의 분포는 $r_s$의 분포보다 더 빨리 정규분포에 수렴한다.

* 독립성 검정에서 두 통계량에 기초한 점근효율은 같다.

* 독립성 검정에서 $\hat{\tau}$와 $r_s$의 값은 달라도 귀무가설을 기각 또는 채택하는 결정은 거의 동일하다.

* $\hat{\tau}$는 모수 $\tau = \pi_c - \pi_d$의 추정량이지만, $r_s$에 대응되는 모수는 없다.

## Robust Methods
Robust 모수 추정은 자료에 이상치 (outlier)가 있는 경우와 자료에서 이상치를 제거한 경우의 모수 추정 결과가 크게 변하지 않는 추정 방법을 의미한다. 예를 들어 표본 평균이나 표본 표준편차는 robust 추정량이 되지 않는다. 반면 중간값 (median)은 robust 추정량이다.

### Three-sigma Rule
**Three-sigma 규칙**이란, 평균에서 양쪽으로 3표준편차의 범위에 거의 자료들(99.7%)이 들어간다는 것을 말한다. 이를 이용하여, $\vert \frac{x_i- \bar{x}}{s} \vert > 3$인 $x_i$를 이상치로 정의하고, 제거할 수 있다.

하지만, Three-sigma 규칙은 다음과 같은 문제점들이 있다.

* 자료의 수가 많은 경우 이상치가 아니지만 앞의 Three-sigma 규칙을 적용하면 이상치로 잘못 판단될 수 있다.

* 자료의 수가 적은 경우 이상치를 찾아내지 못할 수 있다.

* 여러 개의 이상치가 존채할 때, 표본표준편차 값이 켜저 규칙이 좋지 않을 수 있다.

이에 대한 해결책 중 하나로, $t_i = \vert \frac{x_i- \bar{x}}{s} \vert$ 대신 다음 값을 사용할 수 있다.

$$
t_i' = \frac{x_i - \text{median}(x)}{\text{MADN}(x)}
$$

여기서 $\text{MADN}(x) = \text{MAD}(x) / 0.6745$ 이고, $\text{MAD}(x) = \text{median}(\vert x_i - \text{median}(x)\vert)$이다. 이렇게 계산된 $t_i'$을 이용하여 이상치를 판단할 수 있다.

> Sketch of Proof)
> $$
> \sqrt{\frac{1}{n}\sum (x_i - \bar{X})^2} \approx \sqrt{\text{median} \left((x_i - \bar{X})^2 \right)} = \text{median}(\vert x_i - \bar{X} \vert) \approx \text{median}(\vert x_i - \text{median}(x) \vert)
> $$
>
> and $E(\text{MAD}(x))_\text{Normal}=0.6745\sigma$.
