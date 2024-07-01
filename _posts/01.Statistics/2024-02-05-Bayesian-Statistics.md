---
title: Bayesian Statistics
author: rdh
date: 2024-02-05 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [bayesian statistics, statistics]
math: true
---
## Bayesian Inference
**베이지안 추론 (Bayesian Inference)**은 통계적 추론의 한 방법으로, 추론해야 하는 대상의 사전 확률에서 데이터 관측을 통해 해당 대상의 사후 확률을 업데이트하여 추론하는 방법이다. 이는 베이즈 확률론을 기반으로 하며, 이는 추론하는 대상을 확률변수로 보아 그 변수의 확률분포를 추정하는 것을 의미한다.

### Frequentist Approach vs. Bayesian Approach
이러한 베이지안 접근법과 상반되는 접근법은 **빈도주의 (Frequentist approach)**로써, 지금껏 우리가 정리해온 방식이다. 빈도주의에서는 모수를 이용해 확률분포를 정의할 수 있을 때의 경우로 한정해서 설명하며, 확률변수가 특정한 분포를 따른다고 가정하고 그 분포의 모수를 추정한다. 이 때, 모수는 고정된 상수이다.

빈도주의에서 확률은 ‘무한히 많은 시행’에서의 상대적인 빈도로 정의된다. 통계적 추론은 모수를 추정하는 것에 목적을 둔다.

반면 베이지안 접근법의 경우, 자료가 특정한 분포에서 나왔다고 할때, 그 분포의 모수를 고정된 상수가 아니라 확률변수로 가정한다. 즉, 모수도 분포를 가지는 것으로 간주한다.

확률을 빈도나 어떤 시스템의 물리적 속성으로 여기는 빈도주의와는 달리, 베이지안들은 주관주의 확률이론에 따라 확률을 어떤 사람이 특정한 순간에 주어진 명제나 사건에 대해 갖는 믿음의 정도(degree of belief)로 정의한다.

따라서 모수의 분포를 추정할 때 현재 관찰된 자료뿐만 아니라 이전의 자료나 연구자의 믿음 등도 고려되며, 새로운 자료가 수집되면 모수에 대한 추정이 업데이트 된다.

> 잘 생각해보면, 현실에서 확률에 따른 의사결정을 진행하는 과정은 대부분의 경우 빈도주의가 아니라 베이지안 접근법을 따른다. 즉, 의사결정자의 주관적인 확률 이론과 믿음에 의해 결정되는 경우가 많다. 본인은 부정하고 싶겠지만...
{: .prompt-tip}

예를 들어, 명제 "동전 하나를 던졌을 때 앞면이 나올 확률이 50퍼센트이다."를 두 관점으로 보면 다음과 같다.

* 빈도주의: 동전 하나 던지기를 수천, 수만번 하면 그중에 50퍼센트는 앞면이 나오고, 50퍼센트는 뒷면이 나온다.
    * 객관적 확률로 해석
* 베이지안: 동전 하나 던지기의 결과가 앞면이 나올 것이라는 확신은 50퍼센트이다.
    * 주관적 확률로 해석

### Elements Required for Bayesian Inference
베이지안 추론을 위해서는 세 가지 요소가 필요하다.

* **사전분포 (Prior distribution)**: 모수 $\theta$의 분포로 자료를 보기 전 분석자의 $\theta$에 관한 정보(불확실성의 정도)를 나타낸다. $\pi(\theta)$로 나타낸다.
    * 보통 과거 정보를 사전분포로 삼는다.

* **확률모형 (Probability model)**: 주어진 $\theta$에 대해, 데이터 $x$의 분포에 관한 모형이다. $x \mid \theta \sim f(x \mid \theta)$ 또는 $\pi(x \mid \theta)$로 나타낸다.

* **사후분포 (Posterior distribution)**: 데이터 $x$가 주어졌을 때, $\theta$의 확률분포로, 데이터를 본 후의 분석자의 $\theta$에 대한 불확실성을 나타낸다. $\pi(\theta \mid x)$로 나타낸다.
    * 사후분포의 평균 (Posterior mean), 중앙값 (Posterior median), 최빈값 (Maximum a Posteriori, MAP) 등을 모수 $\theta$의 베이지안 추정값으로 사용할 수 있다.

## Posterior Distribution

베이지안 추론에서 확률모형과 사후분포는 조건부 확률분포로 해석한다. 즉, $f(x \mid \theta) = f(x, \theta) / \pi(\theta)$, $f(\theta \mid x) = f(x, \theta) / f(x)$ 이다.

베이즈 정리를 이용하면, 사후확률은 다음과 같다 (여기서는 $f$와 $\pi$를 혼용해서 사용한다).

$$
\pi(\theta \mid x) = f(x \mid \theta)\pi(\theta) / f(x) \propto f(x \mid \theta)\pi(\theta)
$$

> 즉, 사후분포는 가능도와 사전분포의 곱에 비례한다.
{: .prompt-info}

### Example for Normal Distribution
분산 $\sigma^2$가 알려져 있을 때, 평균 $\theta$, 분산 $\sigma^2$인 정규분포를 생각해 보자.

$$
f(x \mid \theta, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(x-\theta)^2}{2\sigma^2} \right), \quad -\infty < \theta < \infty
$$

모수 $\theta$의 사전분포도 정규분포로 가정한다.

$$
\theta \sim N(m, s^2)
$$

즉, 자료 관측 이전에 $\theta$가 대략 $m$이라 믿으며, 그 불확실성이 대략 $s$만큼인 정규분포를 따른다고 믿는 것을 뜻한다.

데이터 $X = (X_1, X_2, \ldots, X_n)$가 관측되었다고 할 때 ($n$개의 관측값), 베이즈정리에 의한 사후분포는

$$
\pi(\theta | x) \propto \frac{1}{\sqrt{2 \pi s^2}} \exp \left\{ - \frac{(\theta - m)^2}{2 s^2} \right\} \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \right)^n \exp \left\{ - \sum_{i=1}^{n} \frac{(X_i - \theta)^2}{2 \sigma^2} \right\}
$$

이 식을 정리하면

$$
\theta | X \sim N \left( \frac{\frac{\bar{X}}{\sigma^2 / n} + \frac{m}{s^2}}{\frac{1}{\sigma^2 / n} + \frac{1}{s^2}}, \left( \frac{1}{\sigma^2 / n} + \frac{1}{s^2} \right)^{-1} \right)
$$

사후분포의 평균 $\frac{\frac{\bar{X}}{\sigma^2 / n} + \frac{m}{s^2}}{\frac{1}{\sigma^2 / n} + \frac{1}{s^2}}$은 데이터의 평균 (이 경우 $\bar{X}$)과 사전분포의 평균 ($m$)의 가중평균으로 볼 수 있다.

> 데이터의 수가 커질수록, 사후분포의 평균은 데이터의 평균에 가까워지고, 사후분포의 산포는 작아진다. 즉, 이는 $\theta$가 데이터의 평균에 가까운 값을 가질 것이라는 믿음이 더 커진 것이라 생각할 수 있다.
{: .prompt-info}

## Bayesian Decision Theory
**베이지안 결정이론 (Bayesian Decision Theory)**이란, 에러를 최소화하고 가장 위험이 작은 결정을 내리는 원칙으로, 에러마다 다른 가중치를 주는 손실함수를 고려하여 ‘기대손실’이 가장 작게끔 하도록 결정을 내리는 원칙을 뜻한다.

> 베이지안 결정이론은 패턴 분류 문제에 대한 기본적인 통계학적 접근방법으로 볼 수 있다.
{: .prompt-info}

베이지안 결정이론에 의한 베이지안 추정량의 예로, 제곱손실함수 (quadratic loss function)의 기댓값을 최소화하는 추정량은 사후분포의 평균 (posterior mean) 이다.

### Example for Bayesian Decision Theory
병원에서 암을 진단하기 위하여 어떤 환자의 X-Ray 사진을 찍었다고 하자. 이 사진을 토대로 그 환자가 암에 걸렸는지 아닌지를 확인하는데에 결정이론이 사용될 수 있다.

X-Ray 결과를 $x$, 환자의 암 여부를 $t$라고 생각하자. $t$가 $C_1$인 경우를 암, $C_2$인 경우를 암이 아닌 경우로 본다. 우리는 X-Ray 결과 $x$를 토대로 그 환자의 암 발병 여부를 정하게 되므로 결국 관심있는 확률은 $p(C_k \mid x)$가 된다.

베이즈 정리를 적용하면 아래의 식이 성립한다.

$$
p(C_k \mid x) = \frac{p(x \mid C_k)p(C_k)}{p(x)}
$$

이 때, $p(C_k)$는 클래스 $C_k$의 사전확률밀도함수, $p(C_k \mid x)$는
사후확률밀도함수가 된다.

이 결정에서의 목적은 실제와는 다른 선택을 할 가능성을 줄이는 데에 있다. 즉, 암이 아닌데 암이라고 판정하거나 암인데 암이 아니라고 판정할 가능성을 줄이는 것이다.

모든 $x$에 대하여 이 결과에 따라 판정을 특정한 경우로 할당하는 규칙이 필요하다. 이러한 규칙은 전체입력공간을 각 경우마다 각각의 결정구역 $R_k$으로 나눈다. 즉 $x \in R_1$인 경우에는 $C_1$이라고 판정을 하는 식이다.

오분류할 가능성을 $p(\text{mistake})$라고 하면, 아래와 같이 나타낼 수 있다.

$$
\begin{aligned}
p(\text{mistake}) &= p(x \in R_1, C_2) + p(x \in R_2, C_1)\\
&= \int_{R_1} p(x, C_2)dx + \int_{R_2} p(x, C_1)dx
\end{aligned}
$$

즉, 위의 $p(\text{mistake})$를 최소화하는 $R_1$과 $R_2$를 찾는 문제가 된다.

그런데 똑같은 오분류라도 실제 상황에서는 암이 아닌데 암인 것으로 진단한 경우보다 암이 맞는데 암이 아닌 것으로 진단한 경우가 훨씬 심각한 문제를 초래하게 된다. 따라서 결정을 내림에 있어서 이러한 정보를 반영하여 생각해
볼 필요가 있다. 예를 들어 위에서 언급한 후자의 경우에 10000배의 패널티를 주고 의사결정과정을 진행할 수 있다.

이 경우에는 행이 각각 실제 암, 실제 암이 아님, 열이 암으로 판정, 암이 아님으로 판정이라고 할 때, $$\begin{bmatrix} 0 & 10000 \\ 1 & 0 \end{bmatrix}$$ 가 손실함수를 나타내는 손실행렬이 된다.

최종적으로 문제는 손실함수를 고려한 아래 기대손실을 최소화시키는 결정을 내리는 것으로 바뀌게 된다.

$$
\mathbb{E}(L) = \sum_k \sum_j \int_{R_j} L_{kj} p(x, C_k) dx
$$

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
