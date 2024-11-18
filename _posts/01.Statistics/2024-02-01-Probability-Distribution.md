---
title: Probability Distribution
author: rdh
date: 2024-02-01 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [probability distribution, statistics]
math: true
---

## Probability
확률(Probability)이란, 어떤 사건(event)이 일어날 가능성을 나타내는 개념으로 사건 A가 일어날 확률을 $P(A)$로 나타낸다.

* $P(A)$ = 사건 A에 속하는 원소의 개수 / 표본공간 전체의 원소의 개수

이 때, 사건 A는 어떤 시행(Experiment, E)에서 나오는 가능한 모든 결과(outcome)들을 모아놓은 집합인 표본공간(Sample space, S)의 부분집합이다.

### Properties of Probability

* $P(\varnothing) = 0$
* $A \subseteq B \implies P(A) \leq P(B)$
* $0 \leq P(A) \leq 1$
* $P(A^c) = 1 - P(A)$
* $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### Conditional Probability
사건 A가 주어졌을 때 사건 B의 조건부확률(Conditional Probability)은 $P(B\vert A)$로 나타내고 $P(A) > 0$이라는 가정 하에 다음과 같이 정의된다.

$$
P(B\vert A) = \frac{P(A \cap B)}{P(A)}
$$

즉, 사건 A를 축소된 새로운 표본공간으로 간주했을 때, 사건 B가 일어날 확률을 말한다.

#### Law of Total Probability
표본공간 $S$의 분할 $\\{A_1, \ldots, A_n\\}$을 생각하자. 표본공간의 분할 (partition)은 다음을 만족한다.
$\forall A_i \cap A_j = \emptyset \ (i \neq j), \ A_1 \cup A_2 \cup \cdots \cup A_n = S$
이때, 전확률공식(law of total probability)는 다음과 같다.

$$
P(B) = P(B \vert A_1)P(A_1) + \cdots + P(B \vert A_n)P(A_n)
$$

### Independence
사건 A가 일어났다고 하더라도 사건 B가 일어날 확률에 아무런 영향을 미치지 않는 경우, 두 개의 사건 A와 B는 서로 독립(independent)이라고 한다.

A와 B는 서로 독립인 경우, $P(B \vert A) = P(B)$ 또는 $P(A \cup B) = P(A)P(B)$가 성립한다.

두 사건 A와 B가 독립이 아니면 종속(dependent)이라고 한다.

* $A \cup B = \emptyset$인 두 사건 A와 B는 서로 배반(mutually disjoint), 즉 두 사건이 동시에 일어날 수 없음을 의미하고 A와 B는 종속 사건이다.
* $A, B$가 독립 사건이면, $A^C, B$, $A, B^C$도 독립 사건이다.

### Bayes Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

### Random Variable
확률 변수(random variable)는 표본공간의 각 원소를 하나의 실수로 대응하는 함수를 말한다.

$$
c \in S, \quad X(c)=x \in \mathbb{R}
$$

* 확률 변수 X의 값이 $\mathcal{B}$에 속할 확률: $P(X \in \mathcal{B}) = P(c \in S \mid X(c) \in \mathcal{B})$

### Probability Distribution
확률변수 X의 확률분포 (probability distribution)란 확률변수 X가 가질 수 있는 값과 해당하는 확률에 대해 나타낸 것으로, 확률을 계산 할 수 있는 정보를 제공한다.

#### Discrete Random Variable
이산확률변수(Discrete Random Variable)는 X가 취할 수 있는 값이 $x_1, x_2, x_3, \dots$와 같이 이산 일 때: 해당 값과 대응하는 확률을 제공한다.

확률분포는 다음과 같은 확률질량함수 (probability mass function, pmf) $p(x)$로 표현한다.

$$
p(x) = P(X = x) =
\begin{cases} 
P(X = x_i) & \text{if } x = x_i \; (i = 1,2, \ldots) \\
0 & \text{otherwise}
\end{cases}
$$

* $0 \leq p(x) \leq 1$
* $\sum_{\text{all } x} p(x) = 1$
* $P(a < X \leq b) = \sum_{a < x \leq b} p(x)$

#### Continous Random Variable
연속확률변수(Continous Random Variable)는 X의 취할 수 있는 값이 셀 수 없이 많을 때: 특정 구간에 속하는 확률을 계산할 수 있는 정보를 제공한다.

확률분포는 확률밀도함수 (probability density function, pdf) $f(x)$ 를 도입하여 X의 값이 $a \leq X \leq b$일 확률로 표현한다.

$$
P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx
$$

* $f(x) \geq 0$
* $\int_{-\infty}^{\infty} f(x) \, dx = 1$
* 연속확률변수의 한 점에서의 확률은 0이다: $P(X=a)=0$

#### Cumulative Distribution Function (CDF)
Cumulative Distribution Function (CDF)는 pmf, pdf 외에 확률분포를 나타내는 또 다른 함수로 다음과 같이 정의된다. (이산확률변수, 연속확률변수에 상관없음)

$$
F_X(x) = P(X \leq x)
$$

* Non-decreasing 함수
* 연속확률변수의 경우: $\frac{d}{dx}F(x)=f(x)$

### Expectation
Expectation은 확률변수 X의 중심을 나타내는 값으로 평균(mean)으로도 부른다.

$$
\mu = \mathbb{E}(X) = 
\begin{cases} 
\sum_{x} x \, p(x) & \text{(이산확률변수)} \\
\int_{-\infty}^{\infty} x \, f(x) \, dx & \text{(연속확률변수)}
\end{cases}
$$

* $\mathbb{E}(X)$: 1st moment, 중심에 대한 정보
* $\mathbb{E}(X^2)$: 2nd moment, 흩어짐에 대한 정보
* $\mathbb{E}((X - \mu)^2)$: 2nd centered moment
* $\mathbb{E}(X^3)$: 3rd moment, symmetric 정보 (skewness, 왜도)
* $\mathbb{E}(X^4)$: 4th moment, tail information (kurtosis, 첨도)

확률변수 X의 함수 $g(X)$의 기대값은 다음과 같다.

$$
\mu = \mathbb{E}(g(X)) = 
\begin{cases} 
\sum_{x} g(x) \, p(x) & \text{(이산확률변수)} \\
\int_{-\infty}^{\infty} g(x) \, f(x) \, dx & \text{(연속확률변수)}
\end{cases}
$$

기대값은 선형성을 갖는다.
* $\mathbb{E}(aX+b) = a\mathbb{E}(X)+b$
* $\mathbb{E}(ag(X)+bh(X)) = a\mathbb{E}(g(X))+b\mathbb{E}(h(X))$

### Variance and Standard Deviation
X의 평균을 $\mu$라고 하자.

* 분산(variance)

    $$
    \text{Var}(X) = \mathbb{E}((X - \mu)^2) = 
        \begin{cases}
        \sum_x (x - \mu)^2 p(x) & (\text{이산확률변수}) \\
        \int_{-\infty}^{\infty} (x - \mu)^2 f(x) \, dx & (\text{연속확률변수})
        \end{cases}
    $$

* 표준편차(standard deviation)

    $$
    \text{sd}(X) = \sqrt{\text{Var}(X)}
    $$

다음 성질을 만족한다.
* $\text{Var}(X) = \mathbb{E}(X^2) - [\mathbb{E}(X)]^2$
* $\text{Var}(aX + b) = a^2 \text{Var}(X)$

## Examples of Probability Distribution
### Bernoulli Distribution
베르누이 시행 (Bernoulli trial)은 실험의 결과가 두 가지 중의 하나로 나오는 시행이다. 즉, 표본 공간 $S = \\{\text{성공}(s), \text{실패}(f)\\}$이고, 성공 확률 $p=P(\\{s\\})$이다.

이 때, 베르누이 확률변수 (Bernoulli random variable)는 베르누이 시행의 결과를 0 또는 1의 값으로 대응시키는 확률변수를 말한다. 즉, $X(s)=1, X(f)=0$인 확률변수이다.

베르누이 확률변수의 확률분포를 베르누이 분포(Bernoulli distribution)라 하고, $X \sim Ber(p)$으로 나타낸다.

* $p(x) = p^x (1 - p)^{1-x}, \quad x = 0, 1$
* $\mathbb{E}(X) = p$
* $\text{Var}(X) = \mathbb{E}(X^2) - [\mathbb{E}(X)]^2 = p(1 - p)$

### Binomial Distribution
이항 분포 (Binomial Distribution)는 베르누이 시행을 n번 독립적으로 시행할 때 성공횟수의 분포로, $X \sim B(n,p)$ 또는 $Bin(n,p)$로 나타낸다.

* $p(x) = \binom{n}{x} p^x (1-p)^{n-x}, \quad x = 0, \ldots, n$
* $n=1$이면, 베르누이 분포
* $\mathbb{E}(X) = np$
* $\text{Var}(X) = np(1 - p)$

### Poisson Distribution
포아송 분포 (Poisson Distribution)는 일정 기간 또는 특정 공간상에서 일어나는 독립적인 사건들의 횟수를 모형화 한 분포로, $X \sim Poi(\lambda)$로 나타낸다.

* $p(x) = \frac{\lambda^x}{x!} e^{-x}, \quad x = 0, \ldots, \lambda>0$
* $\mathbb{E}(X) = \lambda$
* $\text{Var}(X) = \lambda$
* $X_1 \sim Poi(\lambda_1)$, $X_2 \sim Poi(\lambda_2)$ 일 때, $X_1 + X_2 \sim Poi(\lambda_1 + \lambda_2)$

### Uniform Distribution
확률변수 X가 a와 b 사이에서 같은 정도로 값을 가질 때, 균등분포 (Uniform Distribution)를 따른다고 하며, $X \sim Uniform(a,b)$로 나타낸다.

* $f(x) = \frac{1}{b-a}, \quad a < x < b$
* $\mathbb{E}(X) = \frac{a+b}{2}$
* $\text{Var}(X) = \frac{(b-a)^2}{12}$

### Beta Distribution
베타 분포 (Beta Distribution)는 연속확률분포 중의 하나로 $0 \leq X \leq 1$인 확률변수가 다음의 확률밀도함수를 가지는 경우이다.

$$
f(x) = f(x \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}, \quad x \in [0,1], \, \alpha > 0, \beta > 0.
$$

$X \sim Beta(\alpha, \beta)$로 나타낸다.

* $B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}$는 정규화 상수(normalizing constant)라고 한다.
  * $\Gamma(x)$: 감마함수
* $\mathbb{E}(X) = \frac{\alpha}{\alpha + \beta}$
* $\alpha = \beta = 1$이면, 베타분포는 균일분포와 같다.

### Exponential Distribution
지수 분포 (Exponential Distribution)는 하나의 사건이 일어난 후 독립인 그 다음 사건이 일어날 때까지 기다리는 시간 (waiting time)을 모형화 한 분포로, $X \sim Exp(\lambda)$로 나타낸다.

* $f(x) = \lambda \exp(-\lambda x), \quad x>0$
  * $\lambda$: rate parameter
* $f(x) = \frac{1}{\rho} \exp(- x / \rho), \quad x>0$
  * $\rho$: scale parameter
* $\mathbb{E}(X) = \frac{1}{\lambda} = \rho$
* $\text{Var}(X) = \frac{1}{\lambda^2} = \rho^2$
* Memoryless Property: $P(X>s+t \mid X>s) = P(X>t), \, s,t>0$

### Normal Distribution
정규 분포 (Normal Distribution)는 가우스(Gauss, 1777-1855)에 의해 제시된 분포로서 Gaussian distribution라고도 불린다.

물리학 실험 등에서 오차에 대한 확률분포를 연구하는 과정에서 발견된 연속확률분포로, 통계학 초기 발전 단계에서 모든 자료의 히스토그램이
가우스분포의 형태와 유사하지 않으면 비정상적인 자료라고
믿어서 "정규(normal)"라는 이름이 붙게 되었다.

$X \sim N(\mu, \sigma^2)$로 나타내며, 다음과 같은 확률밀도함수를 갖는다.

$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^{2}}}exp(-\frac{(x-\mu)^{2}}{2\sigma^{2}}), \quad -\infty<x<\infty, \, \sigma>0
$$

* $\mu$: 평균, $\sigma^2$: 분산
  * $\tau = 1/\sigma^2$: precision
* $X \sim N(\mu, \sigma^2)$ 일 때, $aX+b \sim N(a\mu+b, a^2\sigma^2)$

#### Standard Normal Distribution
평균이 0이고 표준편차가 1인 정규분포를 표준정규분포 (standard normal distribution)라고 하며, 보통 $Z$로 표기한다.

* 표준화(standardization): $X \sim N(\mu, \sigma^2)$ 일 때, $Z = \frac{X-\mu}{\sigma} \sim N(0,1)$