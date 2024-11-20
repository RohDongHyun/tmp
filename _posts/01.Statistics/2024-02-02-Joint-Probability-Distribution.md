---
title: Joint Probability Distribution
author: rdh
date: 2024-02-02 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [joint probability distribution, statistics]
math: true
---
## Joint Probability Distribution
### Joint Probability Distribution
A **joint probability distribution** represents the probabilities of all possible pairs of values that two random variables can take.

1. Discrete joint PMF

    $$
    p(x,y) = P(X=x, Y=y)
    $$

    * $0 \leq p(x,y) \leq 1$
    * $\sum_x\sum_y p(x,y) = 1$
    * $P(a<X\leq b, c<Y\leq d) = \sum_{a<x\leq b}\sum_{c<y\leq d}p(x,y)$

2. Continuous joint PDF

    $$
    P(a < X \leq b, c < Y \leq d) = \int_a^b \int_c^d f(x, y) \, dy \, dx
    $$

    * $f(x, y) \geq 0$
    * $\int \int f(x, y) \, dx \, dy = 1$
    * $P(a < X \leq b, c < Y \leq d) = \int_c^d \int_a^b f(x, y) \, dx \, dy$


* $E[g(X, Y)] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x, y) f(x, y) \, dx \, dy$

* $E[ag(X, Y) + bh(X, Y)] = a E[g(X, Y)] + b E[h(X, Y)]$


### Marginal PDF
The **marginal PDF** is defined as follows:

* $p_X(x) = \sum_y p(x, y)$
* $f_X(x) = \int f(x, y) \, dy$

Two random variables $X$ and $Y$ are independent if the following condition is satisfied:

* Discrete: $p_{X,Y}(x,y) = p_X(x) p_Y(y)$
* Continuous: $f_{X,Y}(x,y) = f_X(x) f_Y(y)$
  * If $X$ and $Y$ are independent, $E(XY) = E(X)E(Y)$

### Covariance and Correlation Coefficient
* **Covariance**

    $$
    Cov(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E(XY) - \mu_X \mu_Y = E(XY) - E(X)E(Y)
    $$

* **Correlation coefficient** - the linear relationship between two variables

    $$
    Corr(X, Y) = \rho_{XY} = \frac{Cov(X, Y)}{sd(X) sd(Y)}
    $$

The following properties hold for the random variables $X$ and $Y$:

* $Cov(aX + b, cY + d) = ac \, Cov(X, Y)$

* $Corr(aX + b, cY + d) = sign(ac) \, Corr(X, Y)$

* $Var(X \pm Y) = Var(X) + Var(Y) \pm 2 \, Cov(X, Y)$

* $Var(aX + bY) = a^2 \, Var(X) + b^2 \, Var(Y) + 2ab \, Cov(X, Y)$

* $-1 \leq \rho \leq 1$

* If $Y = a + bX$, then $\rho = \pm 1$.

When random variables $X$ and $Y$ are independent,

* $E(XY) = E(X)E(Y)$

* $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$

* $Cov(X, Y) = 0, \, Corr(X, Y) = 0$
  * Note: $Cov(X, Y) = 0$ does not imply that $X$ and $Y$ are independent.

* $Var(X \pm Y) = Var(X) + Var(Y)$

### Conditional Probability Distribution
A **conditional probability distribution** refers to the probability distribution of one random variable given the value of another random variable.

1. Discrete random variables
    For two discrete random variables $X$ and $Y$, the PMF of $Y$ given $X=x$ is:

    $$
    p(y \mid x) = P(Y = y \mid X = x) = \frac{P(X = x, Y = y)}{P(X = x)}
    $$

    Here, $p(y \mid x)$ represents the PMF of $Y$ when $X = x$ is fixed.

2. Continuous random variables
    For two continuous random variables $X$ and $Y$, the PDF of $Y$ given $X=x$ is:

    $$
    f(y \mid x) = \frac{f(x,y)}{f(x)}
    $$

    Here, $f(y \mid x)$ represents the PDF of $Y$ when $X = x$ is fixed.

    * If one variable is discrete and the other is continuous, the conditional distribution can still be well-defined.

### Conditional Independence
Two random variables $X$ and $Y$ are said to be **conditionally independent** given another random variable $Z$ if $X$ and $Y$ are independent when $Z$ is known.

Therefore, for every $x, y, z$, $p(x, y \mid z) = p(x \mid z)p(y \mid z)$ or $f(x, y \mid z) = f(x \mid z) f(y \mid z)$ hold.

* It is denoted as $X \perp Y \mid Z$.

## Random Vectors

A size $p \times 1$ (column) vector $\mathbf{X} = (X_1, \cdots, X_p)^T$, where each element $X_i$ is a random varaible, is called a **random vector**.

The probability distribution of a random vector is referred to as its **joint probability distribution**.

* Joint PMF: $p_{X_1, \cdots, X_p}(x_1, \cdots, x_p)$

* Joint PDF: $f_{X_1, \cdots, X_p}(x_1, \cdots, x_p)$

* Joint CDF: $F_{X_1, \cdots, X_p}(x_1, \cdots, x_p) = P(X_1 \leq x_1, \cdots, X_p \leq x_p)$

### Mean of Random Vectors

$$
E(\mathbf{X}) = E \begin{pmatrix} X_1 \\ \vdots \\ X_p \end{pmatrix} = \begin{pmatrix} E(X_1) \\ \vdots \\ E(X_p) \end{pmatrix} = \begin{pmatrix} \mu_1 \\ \vdots \\ \mu_p \end{pmatrix} = \mu,
$$

* $\mu_i = E(X_i)$

### Covariance Matrix

The **covariance matrix** $\Sigma$ of a random vector $\mathbf{X}$ is defined as:

$$
cov(\mathbf{X}) = E((\mathbf{X} - \mu)(\mathbf{X} - \mu)^T).
$$

Let $var(X_i) = \sigma_i^2, \, cov(X_i, X_j) = \sigma_{ij}$, and $\sigma_{ii} = \sigma_i^2$. Then, the covariance matrix $\Sigma$ can be expressed as:

$$
\Sigma = cov(\mathbf{X}) = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{pmatrix}
$$

* $\Sigma^{-1}$: Precision matrix

### Marginal Probability Distribution
* PMF: $p_{X_i}(x_i) = \sum_{x_j, j \neq i} p(x_1, \cdots, x_p)$

* PDF: $f_{X_i}(x_i) = \int f(x_1, \cdots, x_p) \, dx_1 \cdots dx_{i-1} \, dx_{i+1} \cdots dx_p$

* CDF: $F_{X_i}(x_i) = \lim_{x_j \to \infty, j \neq i} F(x_1, \cdots, x_p)$

### Conditional PMF
For discrete random variables $X_1, \cdots, X_p$, given $X_1 = x_1, \cdots, X_k = x_k$ where $k < p$, the PMF of $X_{k+1}, \cdots, X_p$ is expressed as:

$$
p(x_{k+1}, \cdots, x_p \mid x_1, \cdots, x_k)
$$

$$
= P(X_{k+1} = x_{k+1}, \cdots, X_p = x_p \mid X_1 = x_1, \cdots, X_k = x_k)
$$

$$
= \frac{P(X_1 = x_1, \cdots, X_p = x_p)}{P(X_1 = x_1, \cdots, X_k = x_k)}
$$

* $p(x_{k+1}, \cdots, x_p \mid x_1, \cdots, x_k)$ is the PMF.

### Conditional PDF
For continuous random variables $X_1, \cdots, X_p$, given $X_1 = x_1, \cdots, X_k = x_k$ where $k < p$, the PDF of $X_{k+1}, \cdots, X_p$ is expressed as:

$$
f(x_{k+1}, \cdots, x_p \mid x_1, \cdots, x_k)
$$

$$
= \frac{f(x_1, \cdots, x_p)}{f(x_1, \cdots, x_k)}
$$

* $f(x_{k+1}, \cdots, x_p \mid x_1, \cdots, x_k)$ is the PDF.
* Even when discrete and continuous random variables are mixed, it is still possible to say conditional probability distribution.

### Independence
Random variables $X_1, \cdots, X_p$ are mutually independent if the following condition is satisfied:

For all possible values of $x_1, \cdots, x_p$,

$$
p(x_1, \cdots, x_p) = p_{X_1}(x_1) \cdots p_{X_p}(x_p) \, (\text{discrete})
$$

$$
f(x_1, \cdots, x_p) = f_{X_1}(x_1) \cdots f_{X_p}(x_p) \, (\text{continuous})
$$

* If $X_1, \cdots, X_p$ are mutually independent, then $E(X_1 \cdots X_p) = E(X_1) \cdots E(X_p)$.

## Examples of Multivariate Probability Distribution
### Multinomial Distribution
The **multinomial distribution** models outcomes when there are more than two possible results for each trial in a series of independent experiments.

Let $X_j$ represent the number of times the $j$-th category occurs in $n$ independent trials. That is, $X_j$ is the count of category $j$ out of $n$ trials, such that $X_1 + \dots + X_k = n$.

Let $p_j$ denote the probability of the $j$-th category occurring in a single trial, where $p_1 + \dots + p_k = 1$.

Then, the counts for each category, $\mathbf{X} = (X_1, \dots , X_k)$, follow a multinomial distribution, denoted as $\mathbf{X} \sim \text{Multi}(n, (p_1, \cdots, p_k))$.

* The PMF of multinomial distribution is given by:

    $$
    p(n_1, \cdots, n_k) = p(n_1, \cdots, n_k \mid \mathbf{p})
    $$

    $$
    = P(X_1 = n_1, \cdots, X_k = n_k) = \frac{n!}{n_1! \cdots n_k!} p_1^{n_1} \cdots p_k^{n_k}
    $$

    * $\mathbf{p} = (p_1, \cdots, p_k)$

* The multinomial distribution can be seen as an extension of the binomial distribution. When $k = 2$, the multinomial distribution reduces to the binomial distribution.

* $E(X_j) = np_j$
* $var(X_j) = np_j(1 - p_j)$
* $Cov(X_j, X_{j'}) = -np_jp_{j'}$

### Dirichlet Distribution
The **Dirichlet distribution** is a continuous probability distribution for a vector of random variables $\mathbf{X} = (X_1, \cdots, X_k)$ $(k \geq 2)$, where $0 \leq X_j \leq 1$ and $\sum_{j=1}^k X_j = 1$.

The PDF is defined as:

$$
f(x_1, \cdots, x_k) = f(x_1, \cdots, x_k \mid \alpha) = \frac{1}{B(\alpha)} \prod_{j=1}^k x_j^{\alpha_j - 1},
$$

$$
x_j \in [0, 1], \sum_j x_j = 1, \alpha = (\alpha_1, \cdots, \alpha_k).
$$

$\alpha_j > 0$ are the parameters that define the PDF.

$$
B(\alpha) = \frac{\prod_{j=1}^k \Gamma(\alpha_j)}{\Gamma(\sum_j \alpha_j)} \quad \text{(normalized constant)}
$$

* It is denoted as $\mathbf{X} \sim \text{Dir}(\alpha)$.

* $E(X_j) = \alpha_j / \sum_i \alpha_i$

* When $k = 2$, the Dirichlet distribution reduces to the Beta distribution.

### Multivariate Gaussian Distribution
각 원소가 가우시안 분포 (정규분포)를 따르는 확률벡터의 분포를 다변량 가우시안분포(Multivariate Gaussian Distribution)라고 한다.

* 가우시안 확률벡터 (크기 $p$)의 확률밀도함수는 다음과 같이 정의된다.

    $$
    f(x_1, \cdots, x_p)
    $$

    $$
    = f(x_1, \cdots, x_p \mid \mu, \Sigma)
    $$

    $$
    = (2\pi)^{-\frac{p}{2}} \lvert \Sigma \rvert^{-\frac{1}{2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right),
    $$

    * $\vert \Sigma \vert$는 $\Sigma$의 행렬식 (determinant)이다.

* $\mathbf{X} \sim N_p(\mu, \Sigma)$로 나타낸다.

* 각 원소가 표준정규분포이고 서로 독립이면, $\mathbf{Z} \sim N_p(0, I)$로 표현된다. $I$는 단위행렬 (identity matrix)이다.

* $\Sigma$는 일반적으로 양의 정 부호 행렬 (positive definite matrix)이다.

* 양의 정부호 행렬은 Cholesky decomposition에 의해 $\Sigma = AA^T$로 표현되고 표준정규분포 벡터 $\mathbf{Z}$를 이용하면 $\mathbf{AZ} + \mu \sim N(\mu, \Sigma)$임을 알 수 있다.

* $\sigma_{ij} = E((X_i - \mu_i)(X_j - \mu_j)) = 0$ 이면, 즉 $\Sigma$ 의 $(i,j)$ 원소가 0 이면, $X_i, X_j$ 는 서로 독립이다.

  * 따라서, 서로 독립인 가우시안 확률변수로 이루어진 다변량 가우시안 확률벡터의 공분산 행렬은 대각행렬이다. 즉, $\Sigma = \text{diag}(d_1, \cdots, d_p)$.

* $a_1X_1 + \cdots + a_pX_p$ (적어도 하나의 $a_i$가 0이 아닌 경우)는 가우시안분포(정규분포)를 따른다.

* $X_1, \cdots, X_p$중에 $k \, (k \leq p)$개의 원소를 뽑아 만든 벡터 $\mathbf{X}_s = (X_{i_1}, \cdots, X_{i_k})$도 가우시안분포를 따른다.

* $$\mathbf{X}_s \sim N_s(\mu_s, \Sigma_s), \mu_s = (\mu_{i_1}, \dots, \mu_{i_k})^T, \Sigma_s$$의 $(l, m)$ 원소는 $\sigma_{i_l, i_m}$ 이다.

* $p = 2$인 경우, 이변량 가우시안 (bivariate Gaussian) 분포이며, 확률밀도함수는 다음과 같이 상관계수를 포함한 5개의 모수로 표현 할 수도 있다. 이때, $\sigma_{12} = \rho \sigma_1 \sigma_2$이다.

$$
f(x_1, x_2)
= \frac{1}{2 \pi \sigma_1 \sigma_2 \sqrt{1 - \rho^2}} \exp \left( -\frac{1}{2(1 - \rho^2)} \left[ \frac{(x_1 - \mu_1)^2}{\sigma_1^2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2} - 2 \rho \frac{(x_1 - \mu_1)(x_2 - \mu_2)}{\sigma_1 \sigma_2} \right] \right)
$$


### Partitioned Gaussian Distribution
가우시안 확률벡터의 일부로 만든 벡터의 분포를 분할 가우시안 분포 (Partitioned Gaussian Distribution)라고 하며, 평균벡터와 공분산 행렬은 원 확률벡터의 평균벡터와 공분산행렬을 분할하여 표현할 수 있다.

$\mathbf{X} = (X_1, \cdots, X_p)^T \sim N_p(\mu, \Sigma)$일 때, $$\mathbf{X} = (\mathbf{X}_1^T, \mathbf{X}_2^T)^T$$로 나누어진다고 하자. 편의상 $$\mathbf{X}_1 = (X_1, \cdots, X_m)^T, \mathbf{X}_2 = (X_{m+1}, \cdots, X_p)^T$$라고 하자. 실제로는 순서상관없이 두개의 그룹으로 묶어도 된다.

이때, $$\mathbf{X}_1 \sim N_m(\mu_1, \Sigma_{11}), \, \mu = (\mu_1^T, \mu_2^T)^T,
\Sigma = \begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}$$

#### Conditional Partitioned Gaussian Distribution

$\mathbf{X}_2 = \mathbf{a}$로 주어졌을때 $\mathbf{X}_1$의 조건부 확률분포는

$$
\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{a} \sim N_m \left( \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (\mathbf{a} - \mu_2), \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \right)
$$

$\mathbf{X} = (\mathbf{X}_1, \mathbf{X}_2)$일때, 즉 이변량 가우시안 일때,

$$
\mathbf{X}_1 \mid \mathbf{X}_2 = a \sim N \left( \mu_1 + \frac{\sigma_1}{\sigma_2} \rho (a - \mu_2), (1 - \rho^2) \sigma_1^2 \right)
$$

### Mixure Distribution
여러개의 분포의 선형결합으로 이루어진 분포를 혼합분포(Mixure Distribution)라고 한다.

이산확률분포에서는 $k$개의 이산확률분포의 선형결합으로 이루어진 다음과 같은 확률질량함수를 가진다.

$$
p(x) = w_1 p_1(x) + \cdots + w_k p_k(x) = \sum_{i=1}^k w_i p_i(x)
$$

이때 $p_k(x)$는 확률질량함수이고, $w_i \geq 0, \sum w_i = 1$을 만족한다.

연속확률분포에서는 다음과 같은 확률밀도함수를 가진다.

$$
f(x) = w_1 f_1(x) + \cdots + w_k f_k(x) = \sum_{i=1}^k w_i f_i(x).
$$

#### Gaussian Mixure Distribution
$f_i$들이 가우시안 확률밀도함수인 경우 가우시안 혼합분포(Gaussian Mixure Distribution)라고 한다.

$\phi(x)$를 표준정규분포의 확률밀도함수라고 하자. 즉,

$$
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}.
$$

$X \sim N(\mu, \sigma^2)$인 경우, $X$의 확률밀도함수는 $\frac{1}{\sigma}\phi\left(\frac{X - \mu}{\sigma}\right)$로 표현할 수 있다.

이 경우 $k$개의 구성원을 가지는 가우시안 혼합 분포의 확률밀도함수는 다음과 같이 쓸 수 있다.

$$
f(x) = \sum_{i=1}^k w_i \frac{1}{\sigma_i} \phi\left(\frac{x - \mu_i}{\sigma_i}\right).
$$

* $k = 2$인 경우 $f(x) = w_1 \frac{1}{\sigma_1} \phi\left(\frac{x - \mu_1}{\sigma_1}\right) + (1 - w_1) \frac{1}{\sigma_2} \phi\left(\frac{x - \mu_2}{\sigma_2}\right)$

* $X_1, \cdots, X_n \overset{i.i.d.}{\sim} f(x) = \sum_{i=1}^k w_i \frac{1}{\sigma_i} \phi\left(\frac{x - \mu_i}{\sigma_i}\right)$, 즉, 가우시안 혼합 분포를 따르는 랜덤 추출된 데이터가 있다고 할때, 각 $X_j$는 $w_i$의 확률로 $N(\mu_i, \sigma_i^2)$을 따른다고 해석할 수 있다.

* 군집분석의 모델로 사용할 수 있다.

![](/assets/img/joint-probability-distribution-01.png){: width="650"}

* 왼쪽: 파란선 $N(−1, 1^2)$, 빨간선 $N(2, 2^2)$
* 오른쪽: 파란점선 $0.5 \times N(−1, 1^2)$, 빨간점선 $0.5 \times N(2, 2^2)$ -> 까만선: $0.5 \times N(−1, 1^2) + 0.5 \times N(2, 2^2)$

## Sample Distribution
### Distribution of Sample Mean
표본평균 (sample mean), $\bar{X}$은 표본의 중심경향성을 나타내는 통계량이다.

* 모집단의 평균 (모평균)을 $\mu$라고 하면, 표본평균은 $\mu$의 추정량 (estimator)이다.

* 표본 $\{X_1, X_2, \cdots, X_n\}$가 모평균 $\mu$, 모분산 $\sigma^2$인 모집단에서 추출된 랜덤표본일때,

    $$
    \bar{X} = \frac{1}{n} \sum_{i=1}^n X_i.
    $$

* 무한모집단에서 추출된 랜덤표본일 경우,

    $$
    E(\bar{X}) = \mu, \, Var(\bar{X}) = \frac{\sigma^2}{n}, \, sd(\bar{X}) = \frac{\sigma}{\sqrt{n}}
    $$

* 크기가 $N$인 유한모집단에서 추출된 랜덤표본일 경우,

    $$
    E(\bar{X}) = \mu, \, Var(\bar{X}) = \frac{N - n}{N - 1} \cdot \frac{\sigma^2}{n}.
    $$

### Law of Large Numbers (LLN)
큰 수의 법칙(Law of Large Numbers, LLN)은 표본의 크기 n 이 커질수록 표본평균의 분산은 0에 가까워진다는 것을 말한다. 

표본평균의 기대값은 모평균과 같고, 분산이 작아지므로, $\bar{X}$는 모평균 $\mu$의 근처에 밀집되어 분포함을 알 수 있다. 이러한 결과를 큰수의 법칙이라고 한다.

![](/assets/img/joint-probability-distribution-02.png){: width="650"}

### Central Limit Theorem (CLT)
중심극한정리(Central Limit Theorem, CLT)는 임의의 모집단에 대해 $\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}$의 분포는 표준정규분포 $N(0, 1)$에 근사한다는 것을 말한다.

유한모집단의 경우, 모집단의 크기 $N$과 표본의 크기 $n$이 충분히 크면(단 $N \gg n$) $\frac{N - n}{N - 1}$의 값이 1에 근사하므로, 위의 성질이 성립한다.

중심극한정리를 통해, 모집단의 분포가 어떤 형태이든지 표본의 크기가 크면 표본평균의 분포를 정규분포로 근사할 수 있다.

* 즉, $\bar{X}$의 분포 $\approx N \left( \mu, \frac{\sigma^2}{n} \right)$.

![](/assets/img/joint-probability-distribution-03.png){: width="650"}

#### Normal Approximation Using the Binomial Distribution
$X_1, X_2, \cdots, X_n$이 성공률이 $p$인 베르누이분포를 따르는 무한모집단의 랜덤표본이라고 하자. 이 경우, $S = \sum_{i=1}^n X_i$은 이항분포 $B(n, p)$을 따른다.

중심극한정리를 적용하면, $n$이 충분히 클 때

$$
\frac{S - np}{\sqrt{np(1 - p)}} = \frac{\hat{p} - p}{\sqrt{p(1 - p)/n}}
$$

의 분포는 표준정규분포 $N(0, 1)$에 근사한다.
($\hat{p}$= 베르누이분포의 표본비율 $\frac{S}{n}$)

즉, $n$이 충분히 크고, $np$가 적당한 값이면, $B(n, p)$를 이용하는 확률계산을 $N(np, np(1 - p))$를 이용하여 근사할 수 있다.
