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
**Correlation analysis** is a method used to examine the linear relationship between two variables. 두 변수 간에 어떤 선형적 관계를 가지고 있는 지를 분석하는 방법을 말한다. 

### Correlation Coefficient

The degree of correlation is typically expressed using the **correlation coefficient $\rho$**, defined as:

$$
\rho = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}
$$

Where:
* $-1 \leq \rho \leq 1$,
* $\rho$ close to 1 indicates a strong positive correlation,
* $\rho$ close to -1 indicates a strong negative correlation,
* $\rho=0$ implies no linear correlation.

> although independence also implies $\rho=0$, the reverse is not always true.
{: .prompt-info}

An estimator of $\rho$ is the **Pearson correlation coefficient**, defined as:

$$
r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \cdot \sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$

While $\rho$ is an effective measure of correlation for normally distributed data, its interpretation weakens for nonparametric methods. In such cases, alternative measures with similar properties are used, such as:

* **Kendall's tau**,
* **Spearman's rank correlation coefficient**.

### Kendall's Tau
Kendall's tau is a measure of association between two variables based on the concordance and discordance of paired observations $(X_1, Y_1), \ldots, (X_n, Y_n)$. For each pair $i, j$, it is defined as follows:

* If the signs of $X_i - X_j$ and $Y_i - Y_j$ are the same, the pair is **concordant**, and the probability of such a pair is $\pi_c = \text{P}[(X_i - X_j)(Y_i - Y_j) > 0]$.

* If the signs of $X_i - X_j$ and $Y_i - Y_j$ are different, the pair is **discordant**, and the probability of such a pair is $\pi_d = \text{P}[(X_i - X_j)(Y_i - Y_j) < 0]$.

Kendall's tau is then defined as:

$$
\tau = \pi_c - \pi_d
$$

*i.e.*, the difference between the probabilities of concordant and discordant pairs.

#### Properties of Kendall's Tau

* Since $\pi_c + \pi_d = 1$, $-1 \leq \tau \leq 1$.

* If $X$ and $Y$ are independent, $\pi_c=\pi_d=1/2$, and hence $\tau = 0$.
  * $\tau > 0$: positive association between $X$ and $Y$.
  * $\tau < 0$: negative association between $X$ and $Y$

* If $(X,Y)$ follows a bivariate normal distribution, the relationship between Kendall's tau ($\tau$) and the correlation coefficient ($\rho$) is given by:

$$
\tau = \frac{2}{\pi} \arcsin (\rho).
$$

#### Kendall's Tau Independence Test
To test the independence of $X$ and $Y$, use the null hypothesis $H_0 : \tau = 0$. Alternative hypotheses can be one-sided or two-sided.

1. Compute the counts of concordant ($P$) and discordant ($Q$) pairs:

    $$
    \begin{aligned}
    P &= \text{Number of pairs where} (X_i - X_j)(Y_i - Y_j) > 0, \\
    Q &= \text{Number of pairs where} (X_i - X_j)(Y_i - Y_j) < 0.
    \end{aligned}
    $$

2. Calculate the Kendall test statistic:
   
    $$
    K = P - Q = \sum_{i<j} \text{sign}(X_i - X_j) \text{sign}(Y_i - Y_j),
    $$

    where:

    $$
    \begin{cases} 
		1 & \text{if } t \leq 0, \\ 
        0 & \text{if } t = 0, \\ 
        -1 & \text{if } t > 0. 
    \end{cases}
    $$

3. Testing procedure at significance level $\alpha$:
    * For $H_1 : \tau > 0$, reject $H_0$ if $K \geq k(\alpha, n)$.
    * For $H_1 : \tau < 0$, reject $H_0$ if $K \leq -k(\alpha, n)$.
    * For $H_1 : \tau \neq 0$, reject $H_0$ if $K \geq k(\alpha/2, n)$ or $K \leq -k(\alpha/2, n)$.

Here, $k(\alpha, n)$ is a constant satisfying $P_0[K \geq k(\alpha, n)] = \alpha$.

#### Estimation of Kendall's Tau
The sample correlation coefficient for Kendall's tau is:

$$
\hat{\tau} = \frac{K}{\binom{n}{2}} = \frac{2K}{n(n-1)},
$$

where $K$ is the Kendall test statistic.

### Spearman's Rank Correlation Coefficient
Let $R_i$ and $S_i$ represent the ranks of $X_i$ and $Y_i$, respectively. The Spearman's rank correlation coefficient is defined as:

$$
r_s = \frac{\sum_{i=1}^{n}(R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^{n}(R_i - \bar{R})^2 \cdot \sum_{i=1}^{n}(S_i - \bar{S})^2}}.
$$

> Spearman's rank correlation coefficient is equivalent to the Pearson correlation coefficient applied to ranks.
{: .prompt-info}

#### Spearman Independence Test

1. Compute the ranks $R_i$ and $S_i$ for $X_i$ and $Y_i$ as described above.

2. Calculate Spearman's rank correlation coefficient:

    $$
    r_s = 1 - \frac{6}{n(n^2 - 1)} \sum_{i=1}^{n} (R_i - S_i)^2.
    $$

3. Testing procedure using the critical value $r_s(\alpha, n)$:
    * For $H_1 : \rho_s > 0$, reject $H_0$ if $r_s \geq r_s(\alpha, n)$.
    * For $H_1 : \rho_s < 0$, reject $H_0$ if $r_s \leq -r_s(\alpha, n)$.
    * For $H_1 : \rho_s \neq 0$, reject $H_0$ if $r_s \geq r_s(\alpha/2, n)$ or $r_s \leq -r_s(\alpha/2, n)$.

### Comparison with Kendall's Tau and Spearman's Rank Correlation Coefficient
* Ease of computation: calculating $r_s$ is simpler than calculating $\hat{\tau}$.

* Convergence to normal distribution: the distribution of $\hat{\tau}$ converges to the normal distribution faster than $r_s$.

* Asymptotic efficiency: both statistics have the same asymptotic efficiency for independence tests.

* Decisions on independence: while $\hat{\tau}$ and $r_s$ may yield different values, their conclusions on rejecting or accepting the null hypothesis are almost identical.

* Interpretation: $\hat{\tau}$ is an estimator of $\tau = \pi_c - \pi_d$, but $r_s$ does not correspond to any specific parameter.

## Robust Methods
**Robust parameter estimation** refers to methods where the estimation results remain largely unaffected by the presence of outliers in the data. For instance, the sample mean and sample standard deviation are not robust estimators, whereas the **median** is considered a robust estimator.

### Three-sigma Rule
The **three-sigma rule** states that approximately 99.7% of data values fall within three standard deviations from the mean in a normal distribution. Using this rule, an observation $x_i$ can be defined as an outlier if:

$$
\vert \frac{x_i- \bar{x}}{s} \vert > 3,
$$

where $\bar{x}$ is the sample mean and $s$ is the sample standard deviation.

However, there are several issues with the Three-sigma rule.

* Large sample size: when the sample size is large, observations that are not actual outliers might be incorrectly flagged as outliers.

* Small sample size: when the sample size is small, genuine outliers might not be detected. 자료의 수가 적은 경우 이상치를 찾아내지 못할 수 있다.

* Multiple outliers: if multiple outliers are present, the sample standard deviation $s$ can become inflated, making the rule ineffective.

To address these issues, the following modified statistic can be used instead of $t_i = \vert \frac{x_i- \bar{x}}{s} \vert$:

$$
t_i' = \frac{x_i - \text{median}(x)}{\text{MADN}(x)},
$$

where:
* $\text{MADN}(x) = \text{MAD}(x) / 0.6745$,
* $\text{MAD}(x) = \text{median}(\vert x_i - \text{median}(x)\vert)$. 

This approach is more robust to outliers because it uses the median and median deviation (MAD) instead of the mean and standard deviation, which are sensitive to extreme values.

> Sketch of Proof)
> $$
> \sqrt{\frac{1}{n}\sum (x_i - \bar{X})^2} \approx \sqrt{\text{median} \left((x_i - \bar{X})^2 \right)} = \text{median}(\vert x_i - \bar{X} \vert) \approx \text{median}(\vert x_i - \text{median}(x) \vert)
> $$
>
> and $E(\text{MAD}(x))_\text{Normal}=0.6745\sigma$.
