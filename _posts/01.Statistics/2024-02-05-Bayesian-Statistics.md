---
title: Bayesian Statistics
author: rdh
date: 2024-02-05 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [bayesian statistics, statistics]
math: true
---
## Bayesian Inference
**Bayesian inference** is a method of statistical inference that updates the probability of a hypothesis as new data becomes available. It is based on Bayesian probability, which treats parameters as random variables and seeks to estimate their distribution.

### Frequentist Approach vs. Bayesian Approach
The **frequentiest approach**, which underpins the traditional methods of statistical inference, assumes that parameters are fixed constants and probabilities are defined as the long-run relative frequency of events over repeated trials. Statistical inference in this context involves estimating these fixed parameters under the assumption that the data follows a specific probability distribution.

To summarize,
* Parameters are fixed constants.
* Probability is interpreted as the frequency of an event in a large number of trials.
* Objective is to estimate parameters using methods like maximum likelihood estimation (MLE).

In contrast, the **Bayesian approach** treats parameters as random variables with their own probability distributions. Bayesian inference combines prior knowledge or beliefs about the parameters (expressed as a prior distribution) with observed data to update their belief about the parameters using Bayes' theorem.

To summarize,
* Parameters are random variables with their own distributions.
* Probability is interpreted as a degree of belief in a hypothesis or event, incorporating subjective perspectives.
* The process updates prior beliefs to a posterior distribution based on observed data.

For example, Consider the statement: "The probability of getting heads when flipping a coin is 50%."

* Frequentist interpretation:  
  If the coin is flipped thousands or millions of times, approximately 50% of the flips will results in heads.
    * Objective probability based on the frequency of outcomes over repeated trials.
* Bayesian interpretation:  
  My confidence that the next flip will result in heads is 50%.
    * Subjective probability based on personal belief or prior knowledge about the coin.

> While one might deny it, most real-world decision-making processes are inherently Bayesian, as they involve subjective probabilities, beliefs, and continuous updates based on new evidence.
{: .prompt-tip}

### Elements Required for Bayesian Inference
Bayesian inference requires three key components:

* **Prior distribution ($\pi(\theta)$)**  
  The prior distribution represents the analysts' knowledge or degree of uncertainty about the parameter $\theta$ before observing the data.
    * $\pi(\theta)$: the distribution of $\theta$ based on prior beliefs or historical information.
    * Often, past data or expert knowledge is used to determine the prior distribution.

* **Probability model ($f(x \mid \theta)$ or $\pi(x \mid \theta)$)**  
  The probability model describes the distribution of the data $x$, given the parameter $\theta$.
    * $x \mid \theta \sim f(x \mid \theta)$ or $\pi(x \mid \theta)$: the likelihood of observing the data under the parameter $\theta$.

* **Posterior distribution ($\pi(\theta \mid x)$)**  
  The posterior distribution reflects the updated knowledge about $\theta$ after observing the data $x$. It represents the degree of uncertainty regarding $\theta$ after incorporating the data.
    * $\pi(\theta \mid x)$: the conditional probability distribution of $\theta$ given $x$.
    * The posterior mean, posterior median, and MAP (Maximum a Posteriori) can be used as Bayesian estimators of the parameter $\theta$.

## Posterior Distribution
In Bayesian inference, the likelihood and posterior distributions are interpreted as conditional probability distributions. Specifically:

$$
f(x \mid \theta) = f(x, \theta) / \pi(\theta), \quad f(\theta \mid x) = f(x, \theta) / f(x).
$$

Using Bayes' theorem, the posterior probability is expressed as follows (here, $f$ and $p$ are used interchangeably):

$$
\pi(\theta \mid x) = f(x \mid \theta)\pi(\theta) / f(x) \propto f(x \mid \theta)\pi(\theta).
$$

> The posterior distribution is proportional to the product of the likelihood and the prior distribution.
{: .prompt-info}

### Example for Normal Distribution
Consider a normal distribution with known variance $\sigma^2$ and mean $\theta$. The likelihood is given by:

$$
f(x \mid \theta, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(x-\theta)^2}{2\sigma^2} \right), \quad -\infty < \theta < \infty.
$$

Assume a normal priority distribution for the parameter $\theta$.

$$
\theta \sim N(m, s^2)
$$

This reflects the belief that $\theta$ is approximately $m$, with uncertainty $s^2$.

When data $X = (X_1, X_2, \ldots, X_n)$ is observed ($n$ observations), the posterior distribution, using Bayes' theorem, is:

$$
\pi(\theta | x) \propto \frac{1}{\sqrt{2 \pi s^2}} \exp \left\{ - \frac{(\theta - m)^2}{2 s^2} \right\} \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \right)^n \exp \left\{ - \sum_{i=1}^{n} \frac{(X_i - \theta)^2}{2 \sigma^2} \right\}.
$$

Simplifying this expression results in:

$$
\theta | X \sim N \left( \frac{\frac{\bar{X}}{\sigma^2 / n} + \frac{m}{s^2}}{\frac{1}{\sigma^2 / n} + \frac{1}{s^2}}, \left( \frac{1}{\sigma^2 / n} + \frac{1}{s^2} \right)^{-1} \right),
$$

where $\bar{X}$ is the sample mean.

The posterior mean,

$$
\frac{\frac{\bar{X}}{\sigma^2 / n} + \frac{m}{s^2}}{\frac{1}{\sigma^2 / n} + \frac{1}{s^2}}
$$

is a weighted average of the sample mean $\bar{X}$ and the prior mean $m$, with weights inversely proportional to their variances.

> With more data, the posterior distribution becomes more concentrated around the sample mean, reflecting stronger belief in $\theta$ being close to the observed data's mean.
{: .prompt-info}

## Bayesian Decision Theory
**Bayesian decision theory** is a principle for making decisions that minimize errors and associated risks by considering a **loss function** that assigns different weights to errors. The goal is to make decisions that minimize expected loss.

> Bayesian decision theory provides a fundamental statistical framework for pattern classification problems.
{: .prompt-info}

An example of Bayesian estimators derived from decision theory is the posterior mean, which minimize the expected loss under a quadratic loss function.

### Example for Bayesian Decision Theory
Consider a hospital scenario where an X-ray image is taken to diagnose whether a patient has cancer. Bayesian decision theory can guide the determination of the patient's condition.

* Let $x$ represent the X-ray result.
* Let $t$ represent the true condtion: $t=C_1$ (cancer) or $t=C_2$ (no cancer).

The decision is based on the posterior probability $p(C_k \mid x)$, which can be computed using Bayes' theorem:

$$
p(C_k \mid x) = \frac{p(x \mid C_k)p(C_k)}{p(x)}.
$$

Here, $p(C_k)$ is the prior PDF of class $C_k$, and $p(C_k \mid x)$ is the posterior PDF given $x$.

The objective is to minimize the chance of incorrect decisions, such as:

1. Diagnosing cancer when there is none (false positive).
2. Failing to diagnose cancer when it is present (false negative).

The decision rule is required for every $x$. This rule divide the entire input space into decision regions $R_k$, where:

* $x \in R_1$: decide $C_1$ (cancer).
* $x \in R_2$: decide $C_2$ (no cancer).

Then the probability of misclassification, $p(\text{mistake})$, is:

$$
\begin{aligned}
p(\text{mistake}) &= p(x \in R_1, C_2) + p(x \in R_2, C_1)\\
&= \int_{R_1} p(x, C_2)dx + \int_{R_2} p(x, C_1)dx.
\end{aligned}
$$

Thus, the goal is to find $R_1$ and $R_2$ that minimizes $p(\text{mistake})$.

However, in real-world scenarios, the consequences of errors are not equal. For example, diagnosing cancer incorrectly as no cancer (false negative) has far more serious consequences than vice versa.

To account for this, penalties are incorporated using a loss function. For example, if a false negative has 10000 times the penalty of a false positive, the loss matrix could be:

$$
L=
\begin{bmatrix}
0 & 10000 \\
1 & 0
\end{bmatrix}.
$$

Here, rows correspond to the true condition, and columns correspond to the decision.

Finally, the decision-making process is transformed into minimizing the **expected loss**:

$$
\mathbb{E}(L) = \sum_k \sum_j \int_{R_j} L_{kj} p(x, C_k) dx.
$$

By minimizing $\mathbb{E}(L)$, decisions are optimized to account for both the likelihood of outcomes and the associated risks, ensuring that severe misclassification are appropriated penalized.

## Prior Distribution
사전분포는 데이터를 보기전에 모수 $\theta$의 불확실성을 나타내는 확률분포이다.

### Conjugate Prior
사전분포와 사후분포가 같은 분포족(대개, 지수족)에 속하게 만드는 사전분포를 **켤레사전분포 (Conjugate prior)**라고 한다.

* 예) 데이터가 정규분포를 따를 때, 평균의 사전분포도 정규분포를 따른다고 할 경우, 평균의 사후 분포도 정규분포를 따른다.

### Informative Prior vs. Non-Informative Prior
사전분포는 모수에 대해 구체적 정보를 주는 경우(Informative prior, subjective prior)와 일반적 정보 또는 아예 정보가 없는 경우(non-informative prior, diffuse prior, objective prior)로 나눠서 생각할 수 있다.

Informative prior는 모수에 대한 구체적 정보를 제시하므로, 상대적으로 더 작은 분산을 갖는다. 즉, 모수는 주관적으로 생각한 값에 가까워진다.

* 예) $\mu \sim N(1, 0.1^2)$ vs. $\mu \propto 1 \approx N(1,10^{10})$ or $\mu \sim \text{Unif}(-\infty, \infty)$

### Proper Prior vs. Improper Prior
사전분포를 잘 정의된 확률분포일때 (proper prior)와, 잘 정의되지 않은 확률분포(improper prior), 즉, 적분 또는 합이 유한하지 않을때로 나누어 생각할 수 있다.

예를 들어, $X \sim N(\mu, \sigma^2)$ 인 자료가 있다고 하자. 이때 분산은 알려져 있다고 가정한다. 평균에 대한 특별한 정보(믿음)가 없는경우 평균이 어떤 값을 가지던지 동일한 정도의 정보를 주도록 하고 싶으면 $\mu$가 균일분포 $\text{Unif}(-\infty, \infty)$ 를 따른다고 가정할 수 있다. 다만 평균의 범위가 $(-\infty, \infty)$이므로, 이러한 분포는
적분가능하지 않다.

사전분포가 improper prior이어도 사후분포가 proper prior가 될 수 있다. 다만 항상 되는 것은 아니기 때문에 확인이 필요하다.

### Example: Binomial and Beta Prior Distribution
연속된 $n$번의 독립적 시행에서 각 시행이 확률 $\theta$ (단, $0 \leq \theta \leq 1$)일 때 성공횟수는 이항분포를 따르며, 이항분포의 확률질량함수는 다음과 같다.

$$
f(x \vert \theta) = \frac{n!}{x!(n-x)!} \theta^x (1-\theta)^{n-x}, \quad 0 \leq \theta \leq 1
$$

이 때, 모수 $\theta$에 대한 사전분포로 베타분포를 고려해 보자: $\theta \sim Beta(\alpha, \beta), \alpha, \beta > 0$

베타분포의 확률밀도함수는 다음과 같다.

$$
f(\theta \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}
$$

데이터 $X$가 관측되었다고 할때, 베이즈 정리를 이용하여 $\theta$의 사후분포를 구할 수 있다.

$$
\begin{aligned}
f(\theta \vert X) &\propto Lik(\theta \vert X) \pi(\theta) \\
&\propto \frac{n!}{X!(n-X)!} \theta^X (1-\theta)^{n-X} \cdot \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1} \\
&\propto \theta^{X + \alpha - 1} (1 - \theta)^{n - X + \beta - 1} \\
&\propto \text{pdf of } Beta(\alpha + X, \beta + n - X)
\end{aligned}
$$

따라서, $\theta$의 사후분포는

$$
\theta \mid X \sim Beta(\alpha + X, \beta + n - X)
$$

이항분포인 Likelihood에 대하여 베타사전분포를 사용하면 사후분포 역시 베타분포임을 알 수 있다. 즉, 베타사전분포는 이항가능도함수에 대한 켤레 사전분포가 된다.

데이터 X가 추가(관측)되면서 모수 θ의 분포가 수정(업데이트) 되었다는 것 역시 알 수 있다.

## Naive Bayes Classifier
$N$개의 특성(feature)들을 가지고 $K$개의 클래스 $(C_1, \cdots, C_K)$ 중 하나로 분류시키는 문제를 생각해보자.

이 경우 하나의 관측값은 $N$개의 특성들을 모아놓은 벡터로 생각할 수 있고, 이 벡터를 확률벡터 $\mathbf{X} = (X_1, \cdots, X_N)$라고 본다면, 주어진 문제는 조건부 확률 $P(C_k \vert \mathbf{X})$를 계산하여 그 값이 가장 큰 클래스로 정하는 문제로 볼 수 있다.

즉, $\arg \max_{k \in \{1, \cdots, K\}} P(C_k \vert \mathbf{X})$를 찾는 문제이다.

$P(C_k \vert \mathbf{X}) = f(\mathbf{X})$를 $\mathbf{X}$에 대한 함수로 본다면, 클래스를 아는 여러개의 $\mathbf{X}$들을 관측하여 $f(\mathbf{X})$를 추정하는 문제로 볼 수 있다.

나이브 베이즈는 이러한 방법 중 하나로, 주어진 클래스 상에서 특성들이 $(X_j)$ 서로 독립이라는 가정을 통해 (실제로 독립이 아닐지라도) $P(C_k \vert \mathbf{X})$를 다음과 같이 단순화 시킨다.

$$
\begin{aligned}
P(C_k \vert \mathbf{X}) &= \frac{P(\mathbf{X} \vert C_k)P(C_k)}{P(\mathbf{X})} \quad \text{베이즈 정리 이용}\\
&= \frac{1}{P(\mathbf{X})}P(C_k)P(X_1, X_2, \cdots, X_N \vert C_k)\\
&\approx \frac{1}{P(\mathbf{X})}P(C_k)\prod_{j=1}^{N} P(X_j \vert C_k) \quad \text{독립 가정 (Naive)}
\end{aligned}
$$

$P(C_k \vert \mathbf{X})$에서 $\mathbf{X}$는 주어진것으로 보기때문에, 나이브 베이즈 분류기는 다음과 같다.

$$
\begin{aligned}
& \quad \arg \max_{k \in \{1, \ldots, K\}} P(C_k \vert \mathbf{X}) \\
&= \arg \max_{k \in \{1, \ldots, K\}} \frac{1}{P(\mathbf{X})} P(C_k) \prod_{j=1}^{N} P(X_j \vert C_k) \quad \text{NaiveBayes} \\
&= \arg \max_{k \in \{1, \ldots, K\}} P(C_k) \prod_{j=1}^{N} P(X_j \vert C_k) \\
\end{aligned}
$$

여기서 $P(C_k)$는 k 번째 클래스에 속할 사전확률로 아무 사전 정보가 없다면 $P(C_k) = \frac{1}{K}$로 놓을수 있다. $P(X_j \vert C_k)$는 클래스 k에 속했을때의 j번째 특성 $X_j$의 확률로 데이터를 통해 추정한다.

예를 들어 클래스 $k$에 속하는 $n_k$개의 관측값 $X_1^{(k)}, \ldots, X_{n_k}^{(k)}$이 있을때, j번째 특성값의 데이터는 각 $X_i^{(k)}$의 j번째 원소를 뽑은 $X_{1j}^{(k)}, \ldots, X_{n_k j}^{(k)}$ 들이고, j번째 특성값들이 정규분포를 따른다면, 해당 정규분포의 $\mu_{kj}, \sigma_{kj}^2$는 $X_{1j}^{(k)}, \ldots, X_{n_k j}^{(k)}$를 이용하여 추정하여 사용한다.

$$
\hat{P}(X_j \vert C_k) = \frac{1}{\sqrt{2 \pi \sigma_{kj}^2}} e^{-\frac{1}{2 \sigma_{kj}^2} (X_j - \hat{\mu}_{kj})^2}
$$
