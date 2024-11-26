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

* **Random sample** : a set of independent and identically distributed random variables. The actual values obtained from the sample are called **observations**.

Parameter estimation includes **point estimation** and **interval estimation**.

### Point Estimation
Point estimation involves providing a single estimate for a population parameter calculated from the sample. This estimated value is called the **estimator**.

* Estimator for the population mean : the sample mean $\hat{\mu} = \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$.

* Estimator for the population variance : the sample variance $\hat{\sigma}^2 = S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2$.

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
**Maximum likelihood estimation (MLE)** is a method for estimating the value of a parameter by maximizing the **likelihood function**.

The **likelihood** represents the probability of observing the given data under a specified probability distribution and parameter value. It quantifies how likely a parameter value is given the observed data.

The likelihood function for a parameter $\theta$ is generally expressed as:

$$
L(\theta) = L(\theta \mid X_1, \cdots, X_n) = f(X_1, \cdots, X_n \vert \theta)
$$

where $f$ is the joint probability function (for discrete variables) or joint PDF (for continuous variables) of the random sample $X_1, \cdots, X_n$.

In this context, MLE estimates the parameter by finding the value that maximizes the likelihood function given the observed data.

$$
\hat{\theta}_{MLE} = \arg\max_{\theta \in \Omega} L(\theta \mid X_1, \cdots, X_n)
$$

For example, suppose $X_1, X_2, X_3$ are independent random variables following $Ber(p)$, and the observations are as follows: $X_1 = 1, X_2 = 0, X_3 = 0$. Substituting the observed values into the joint PMF gives:

$$
p_{X_1, X_2, X_3}(1,0,0) = p \times (1-p) \times (1-p).
$$

Thus, the likelihood function for the success probability $p$ is $L(p) = p(1-p)^2$. Therefore, the MLE of $p$ is $\frac{1}{3}$.

#### Log-likelihood
Applying a logarithmic transformation to the likelihood function can simplify the process of finding the MLE:

$$
\hat{\theta} = \arg\max_{\theta \in \Omega} L(\theta) = \arg\max_{\theta \in \Omega} \ell(\theta).
$$

This approach is particularly useful when independent observations are given.

$$
\ell(\theta) = \log(L(\theta)) = \log\left(\prod f(X_i; \theta)\right) = \sum_{i=1}^{n} \log(f(X_i; \theta))
$$

#### Properties of MLE
MLE is an estimator with desirable properties in general, characterized by minimal variance and the fastest convergence rate.

* Consistency: $\hat{\theta} \rightarrow \theta$ in probability.
* Asymptotic normality:
    $$\sqrt{n}(\hat{\theta} - \theta) \rightarrow \mathcal{N}(0, \mathcal{I}^{-1}(\theta))$$ in distribution,
    * where $$\mathcal{I}(\theta) = \mathbb{E} \left[ \left( \frac{d}{d\theta} \log f(X; \theta) \right)^2 \right]$$ is called the Fisher information.

> In other words, asymptotically (as $n \rightarrow \infty$), the MLE follows a normal distribution with mean $\theta$ and variance $\frac{1}{\mathcal{I}(\theta)}$.
{: .prompt-info}

#### CI with MLE
The asymptotic distribution of the MLE can be used to construct a CI for $\theta$.

Given the relationship:

$$
\sqrt{n\mathcal{I}(\theta)}(\hat{\theta} - \theta) \sim \mathcal{N}(0, 1),
$$

the $100(1-\alpha)$% CI for $\theta$ is:

$$
\left( \hat{\theta} - Z_{1-\alpha/2} \frac{1}{\sqrt{n\mathcal{I}(\hat{\theta})}}, \hat{\theta} + Z_{1-\alpha/2} \frac{1}{\sqrt{n\mathcal{I}(\hat{\theta})}} \right).
$$

## Hypothesis Test
**Hypothesis test**, also known as a **test of significance**, is a statistical decision-making process used to determine whether to maintain or reject an existing theory or law when an observation appears to contradict it.

* The hypothesis established to seek refutation (often representing the "current belief") is called the **null hypothesis ($H_0$)**.
* The hypothesis proposed as an alternative to the null hypothesis is called the **alternative hypothesis ($H_1$)**.

The hypothesis test provides a measure of the strength of evidence against the null hypothesis.

> The alternative hypothesis typically represents the claim or assertion that we aim to support.
{: .prompt-info}

### Definitions of Terms

#### Types of Hypotheses

* Simple hypothesis: a hypothesis that specifies a parameter to a single value, such as $\theta = \theta_0 or p = 0.5$ (commonly $H_0$).

* Composite hypothesis: a hypothesis that assumes multiple possible values for the parameter.

* One-sided hypothesis: a hypothesis that specifies a one-sided comparison, such as $\theta > \theta_0$ or $\theta < \theta_0$ (commonly $H_1$).

* Two-sided hypothesis: a hypothesis that specifies a two-sided comparison, such as $\theta \neq \theta_0$ (commonly $H_1$).

#### Types of Errors
* **Type Ⅰ error (false positive)**: an error that occurs when the null hypothesis is true but is incorrectly rejected.

* **Type Ⅱ error (false negative)**: an error that occurs when the null hypothesis is false but is not rejected.

#### Significance Level and Power
* **Significance level ($\alpha$)**: the probability of committing a Type Ⅰ error 
    * For example, $\alpha$ = 0.05 means limiting the probability of rejecting a true null hypothesis to 5% or less.

* **Power of a test**: the probability of rejecting $H_0$ when $H_0$ is false.
    * If the probability of a Type Ⅱ error is $\beta$, the power of the test is $1-\beta$.

#### Test Statistics and Critical Region
* **Test statistic**: a statistic calculated from the sample, used to decide whether to reject $H_0$.

* **Critical region**: the range of values of the test statistic for which $H_0$ is rejected.

#### P-value
* **P-value**: the smallest significance level at which the observed test statistic leads to the rejection of $H_0$. Alternatively, it is the probability of the test statistic falling into the critical region under $H_0$.
    * If the p-value obtained from the test statistic is smaller than the specified significance level, the result is said to be **statistically significant**.

### Procedure of Hypothesis Test
Key considerations in hypothesis testing are as follows:

* Different test statistics result in different tests.

* The form of the critical region is influenced by the alternative hypothesis.

* To compute the p-value or determine the critical region, the distribution of the test statistic under the null hypothesis must be known.

* The case with a higher risk of error is generally designated as a Type Ⅰ error. That is, the null hypothesis typically represents the established or widely accepted belief.

When conducting a hypothesis test, it is essential to first decide which test to use. It is difficult to minimize both Type Ⅰ and Type Ⅱ errors simultaneously, so the usual approach is to control the probability of Type Ⅰ error (significance level) and then select the test that minimizes the probability of a Type Ⅱ error.

> A test that maximizes power (minimizes Type Ⅱ error) for a given significance level is called the most powerful test.
{: .prompt-info}

The procedures of hypothesis testing is as follows:

1. Define the null hypothesis, alternative hypothesis, and significance level.

2. Determine the testing method, then extract samples and calculate the value of the test statistic.

3. Decide whether to reject the null hypothesis and draw a conclusion:
    1. Identify the critical region from the significance level and check whether the test statistic falls within the critical region, or 
    2. Calculate the p-value from the test statistic and compare it with the significance level.

### Test for the Population Mean
When making inferences about the population mean $\mu$, the sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$ is typically used.

1. **When the population variance is known**, use the normal distribution:
    * $H_0$ : $\mu = \mu_0$

    * Test statistic: $Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \overset{H_0}{\sim} N(0,1)$

    * Observed value of the test statistic: $z_0$

    * $P(N(0,1) \leq z_{\alpha}) = \alpha$

    |      $H_1$       |                P-value                |         Critical region          |
    | :--------------: | :-----------------------------------: | :------------------------------: |
    |  $\mu > \mu_0$   |             $P(Z > z_0)$              |        $Z > z_{1-\alpha}$        |
    |  $\mu < \mu_0$   |             $P(Z < z_0)$              |       $Z < -z_{1-\alpha}$        |
    | $\mu \neq \mu_0$ | $P(\vert Z \vert > \vert z_0 \vert )$ | $\vert Z \vert > z_{1-\alpha/2}$ |

2. **When the population variance is unknown**, use the $t$-distribution (assuming the data follow a normal distribution):

    * $H_0$ : $\mu = \mu_0$

    * Test statistic: $T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \overset{H_0}{\sim} t(n-1)$
        * $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$

    * Observed value of the test statistic: $t_0$

    * If $T \sim t(k)$, $P(T \leq t_p(k)) = p$.

    |      $H_1$       |               P-value                |            Critical region            |
    | :--------------: | :----------------------------------: | :-----------------------------------: |
    |  $\mu > \mu_0$   |             $P(T > t_0)$             |        $T > t_{1-\alpha}(n-1)$        |
    |  $\mu < \mu_0$   |             $P(T < t_0)$             |       $T < -t_{1-\alpha}(n-1)$        |
    | $\mu \neq \mu_0$ | $P(\vert T \vert > \vert t_0 \vert)$ | $\vert T \vert > t_{1-\alpha/2}(n-1)$ |

### Test for the Population Variance
Inference about the population variance $\sigma^2$ is typically based on the sample variance $S^2 = \frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X})^2$.

If $X_1, X_2, \ldots , X_n$ are a random sample from a normal distribution $N(\mu, \sigma^2)$, then:

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$

> Inference about the population variance using the chi-squared distribution is highly sensitive to the normality assumption. Ensure the normality assumption is adequately examined before proceeding.
{: .prompt-warning}

* $H_0$ : $\sigma^2 = \sigma^2_0$

* Test statistic : $X^2 = \frac{(n-1)S^2}{\sigma^2_0} \overset{H_0}{\sim} \chi^2(n-1)$

* Observed value of the test statistic: $X^2_0$

* If $X^2 \sim \chi^2(k)$, $P(X^2 \leq \chi^2_p(k)) = p$.

|           $H_1$            |                               P-value                                |                               Critical region                               |
| :------------------------: | :------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
|  $\sigma^2 > \sigma^2_0$   |                           $P(X^2 > X^2_0)$                           |                       $X^2 > \chi^2_{1-\alpha}(n-1)$                        |
|  $\sigma^2 < \sigma^2_0$   |                           $P(X^2 < X^2_0)$                           |                        $X^2 < \chi^2_{\alpha}(n-1)$                         |
| $\sigma^2 \neq \sigma^2_0$ | $2P(X^2 > X^2_0)$ or <br> $2P(X^2 < X^2_0)$ whichever is less than 1 | $$X^2 > \chi^2_{1-\alpha/2}(n-1)$$ or <br> $$X^2 < \chi^2_{\alpha/2}(n-1)$$ |
