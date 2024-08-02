---
title: Markov Chain Monte-Carlo
author: rdh
date: 2024-02-06 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [monte carlo method, inverse cdf, rejection sampling, importance sampling, statistics]
math: true
---
## Approximate Bayesian Method
Bayesian inference를 요약하면 다음과 같다.

* 데이터의 분포에 관한 모형 (likelihood, $L(\theta\mid X)$)와
* 모델을 정하는 모수 (parameter)의 사전분포 (prior distribution, $\pi(\theta)$)를 이용하여
* 모수의 사후확률 (posterior probability, $\pi(\theta\mid X)$)를 구하고,
* 이를 이용하여 모수에 관한 추론을 진행

이 때, likelihood가 복잡하거나, 데이터가 아주 큰 경우에는 사후확률을 계산하는 것이 쉽지 않다.

이에 따라, 여러 **근사 베이지안 방법 (approximate Bayesian method)**들이 제안되었다.

## Markov Chain Monte Carlo (MCMC)
**Markov Chain Monte Carlo (MCMC)**은 근사 베이지안 방법의 일종으로, 다음과 같이 진행된다.

1. 사후분포를 이론적으로 구하는 대신 우리가 원하는 사후분포를 점근분포로 갖는 Markov chain을 만든다.
2. 해당 chain을 충분히 진행히 진행하여 어느정도 수렴이 되면, 이로부터 sample을 추출한다.
3. 이 sample을 사후분포의 sample로 볼 수 있으므로 이를 이용하여 추론을 진행한다.

사후분포의 형태에 따라, **깁스 샘플링 (Gibbs sampling)** 또는 **메트로폴리스-헤이스팅 샘플링 (Metropolis-Hastings sampling)**을 사용한다.

> MCMC 방법은 복잡한 모형인 경우 또는 데이터가 큰 경우 계산속도가 느리다.
{: .prompt-info}

### Markov Chain
$X_t$를 $\pi_t(\cdot)$을 분포로 갖는 시각 $t$에서의 상태 벡터라고 하면, **Markov property**는 '다음 시점의 상태 $X_{t+1}$은 오로지 현재의 상태 $X_t$에만 의존하며, 그 이전에 일어난 사건과는 무관하다'라는 것을 말한다. 즉, 주어진 현재의 상태에 대하여 미래와 과거의 상태는 독립이다.

$$
P(X_{n+1} = x \mid X_n = x_n, \cdots, X_1 = x_1) = P(X_{n+1} = x \mid X_n = x_n)
$$

> Markov property를 무기억성(memorylessness)라고도 한다.
{: .prompt-info}

**Markov chain**이란 여러 가능한 상태(state) 사이에서, 어느 한 상태로부터 다른 상태로의 전이(transition)를 겪는 수학적 시스템을 뜻하며, Markov property를 갖는 확률 변수 $X_1, X_2, \cdots$들의 수열로써 표현된다.

Markove chain에서 각 상태에서 다음 상태로 넘어갈 수 있는 확률를 **전이확률(transition probability)**라고 하며, 전이확률을 matrix 형태로 나타낸 전이행렬(transition matrix) 또는 전이확률에 대한 함수인 전이함수를 이용하여 현재의 상태로부터 그 다음, 다다음 상태의 확률분포를 계속하여 계산하는 것이 가능하다.

이러한 과정을 반복하다 보면 특정 조건 하에서 현재 상태 (state)의 확률분포가 그 전 상태의 확률분포와 같아지는 때가 온다. 이렇게 평형상태에 도달한 확률분포를 **정상분포(stationary distribution)** 또는 **점근분포(limiting distribution)** 라고 하며, 이 분포는 초기값에 의존하지 않는다.

### Procedure of MCMC

1. 점근분포가 $\pi(x)$인 Markov chain $X_0, \cdots, X_t$을 생성한다.
2. $t$를 충분히 크게 만들어, $X_t$가 $\pi(x)$를 따른다고 가정하는데 무리가 없으면, 이후에 생성되는 $X_{t+1}, \cdots, X_{t+m}$을 저장한다.
3. 이렇게 생성된 $X_{t+1}, \cdots, X_{t+m}$을 $\pi(x)$를 따르는 난수로 본다.

> $t$를 충분히 크게 만들어 초기값의 영향을 받지 않게 하는 과정을 burn-in period 라고 한다.
{: .prompt-info}

이제 문제는 이러한 Markov chain을 어떻게 만들 수 있을지가 되며, 이에 대한 하나의 해결책으로 **Metropolis-Hastings algorithm**이 있다.

## Metropolis-Hastings Algorithm
Metropolis-Hastings algorithm 하에서 각 시각 $t$에서 Markov chain의 다음 상태인 $X_{t+1}$은 아래와 같은 과정을 통하여 결정된다. 

1. 어떠한 제안분포(proposal distribution) $q(\cdot\mid X_t)$로부터 후보(candidate point)가 되는 샘플 $Y$를 뽑는다.
    * 제안분포는 현재의 상태인 $X_t$에 의존할 수 있다.
    * 예를 들어, $q(\cdot\mid X_t)$은 평균이 $X_t$이고 고정된 공분산행렬을 가지는 다변량 정규분포일 수 있다.
2. 각 시각 t에 대하여, 마코프 체인의 다음 상태인 $X_{t+1}$을 결정하기위해 다음과 같은 Acceptance ratio를 사용한다.

    $$
    \alpha(X, Y) = \min \left( 1, \frac{\pi(Y)q(X|Y)}{\pi(X)q(Y|X)} \right)
    $$

    후보가 되는 $Y$는 $\alpha(X_t, Y)$의 확률로 다음 상태로 받아들여질지 아닐지 결정된다.
    
    * 위의 식은 마코프 체인의 수렴을 위한 조건중 하나인 정상분포의 존재성을 위한 Detailed Balance 조건으로부터 얻어진 식이다: $\pi(X\mid Y)\pi(Y) = \pi(Y\mid X)\pi(X)$.
    * 제안분포와 관련된 확률식 $\pi(Y\mid X) = q(Y\mid X)\alpha(X, Y)$을 위의 Detailed Balance 조건식에 대입해 $\frac{\pi(Y)q(X\mid Y)}{\pi(X)q(Y\mid X)}$ 항을 얻을 수 있다.

3. $X_{t+1}$의 후보가 되는 $Y$를 $q(Y\mid X_t)$로부터 생성한다. 그 다음 $\alpha(X_t, Y)$의 확률로 다음 상태를 $Y$로 할지를 결정한다.
    * 만약 $Y$가 받아들여지면, 다음 상태는 $X_{t+1} = Y$가 된다.
    * 만약 $Y$가 받아들여지지 않으면, 다음 상태는 $X_{t+1} = X_t$가 된다.

> Metropolis-Hastings algorithm은 원래 분포인 $\pi(\cdot)$을 정확히 몰라도 쓸 수 있다는 장점이 있다. 즉, 정확한 확률분포가 아니라 그에 비례하는 비정규화 분포 (un-normalized distribution)만 알아도 알고리즘을 사용할 수 있다.  
> 하지만, 목표로 하는 $\pi(\cdot)$의 분포를 따르는 $X_t$로 수렴하는데까지 시간이 오래 걸릴 수 있다는 단점이 있다.
{: .prompt-info}

### Example of Metropolis-Hastings Algorithm - Cauchy Model
$Y_1, \cdots, Y_n \sim N(\theta, 1)$ (i.i.d)인 데이터를 생각하자. 이 때, $\theta$의 사전확률로 $\pi_0(\theta) = \frac{1}{\pi(1 + \theta^2)}$를 가정하자 (첫번째 $\pi_0(\cdot)$는 확률밀도함수, 두번째 $\pi$는 원주율). 

이 경우, Bayes theorem에 의해 $\theta$의 사후확률은 다음과 같다.

$$
\begin{aligned}
\pi(\theta \mid Y_1, \cdots , Y_n) &\propto \exp \left( -\frac{\sum_{i=1}^{n} (Y_i - \theta)^2}{2} \right) \times \frac{1}{1 + \theta^2} \\
&\propto \exp \left( -\frac{n(\theta - \bar{Y})^2}{2} \right) \times \frac{1}{1 + \theta^2}
\end{aligned}
$$

이 사후분포의 형태 ($\theta$의 함수로서)는 일반적인 꼴 (아는 분포)이 아니므로, Metropolis-Hastings algorithm을 이용하여 정상분포가 $\pi(\theta \mid Y_1, \cdots , Y_n)$이 되는 Markov chain을 생성하여 sampling을 진행할 수 있다.

우선 제안분포 $q(\theta \mid \theta^\ast)$를 $N(\bar{Y}, \frac{1}{n})$ 으로 정한다. 이 경우에 해당하는 acceptance ratio $\alpha$식을 계산한다.

$$
\begin{aligned}
\alpha(\theta^{(i-1)}, \tilde{\theta}) &= \min \left( 1, \frac{\pi(\tilde{\theta} \mid Y_1, \cdots , Y_n)q(\theta^{(i-1)} \mid \tilde{\theta})}{\pi(\theta^{(i-1)} \mid Y_1, \cdots , Y_n)q(\tilde{\theta} \mid \theta^{(i-1)})} \right) \\
&= \min \left( 1, \frac{1 + (\theta^{(i-1)})^2}{1 + (\tilde{\theta})^2} \right)
\end{aligned}
$$

이 때의 Metropolis-Hastings algorithm은 다음과 같이 동작한다.
  
1. 초기값 $\theta^{(0)}$을 설정한다. 예를 들어, $\theta^{(0)} = 1$. 이 후, $i = 1$부터 $N$까지 다음 단계들을 반복한다.
2. $q(\theta \mid \theta^{(i-1)})$로부터 체인의 다음상태에 대한 후보로 $\tilde{\theta}$를 생성한다.
3. Acceptance ratio $\alpha$를 계산한다.
4. $\text{Unif}(0,1)$을 따르는 $r$을 생성한다.
5. $r < \alpha$이면 $\theta^{(i)} = \tilde{\theta}$, 아니면 $\theta^{(i)} = \theta^{(i-1)}$로 둔다.

> 이렇게 생성한 난수들로 $\theta$의 사후 기댓값, 사후 중앙값, 사후 최빈값 등을 근사적으로 추정할수 있다.
{: .prompt-tip}

## Gibbs Sampling
Gibbs 샘플링은 다변수 확률 분포로부터 샘플을 생성하는 근사 베이지안 방법 중 하나이다. 이를 통해 복잡한 다변수 분포의 샘플링 문제를 각 변수의 조건부 분포로 나눠서 해결할 수 있다.

기본 아이디어는 $p$-차원의 모수의 사후분포로부터 생성하는 Markov chain을 $p$개의 1차원 모수의 조건부 사후분포의 Markov chain으로 만들어 난수를 생성하겠다는 것이다.

예를 들어, 데이터 $Y_1, \cdots, Y_n$이 $N(\mu, \sigma^2)$을 따른다고 했을때 관심모수는 $\Theta = (\mu, \sigma^2)^T$이므로 2차원 모수로 볼 수 있다. 사후확률분포는 $\pi(\Theta \mid Y_1, \cdots, Y_n) = \pi(\mu, \sigma^2 \mid Y_1, \cdots, Y_n)$이다.

Markov chain의 시작점을 $\Theta^{(0)} = (\theta_1^{(0)}, \cdots, \theta_p^{(0)})^T$라 할 때, **Gibbs sampler**는 아래와 같은 알고리즘으로 $\Theta^{(s-1)}$로부터 $\Theta^{(s)}$를 만들어 낸다:

$$
\begin{cases}
\theta_1^{(s)} &\sim \pi(\theta_1 \mid \theta_2^{(s-1)}, \theta_3^{(s-1)}, \cdots, \theta_p^{(s-1)}, \text{Data}) \\
\theta_2^{(s)} &\sim \pi(\theta_2 \mid \theta_1^{(s)}, \theta_3^{(s-1)}, \cdots, \theta_p^{(s-1)}, \text{Data}) \\
&\cdots \\
\theta_p^{(s)} &\sim \pi(\theta_p \mid \theta_1^{(s)}, \theta_2^{(s)}, \cdots, \theta_{p-1}^{(s)}, \text{Data})
\end{cases}
$$

> 이 조건부 분포들을 우리가 아는 경우에는 그냥 sampling을 진행하고, 모르는 경우에는 Metropolis-Hastings algorithm을 통한 sampling을 진행한다.
{: .prompt-info}

Gibbs sampling은 다음과 같은 확률벡터열을 만들어낸다.

$$
\begin{align*}
\Theta^{(1)} &= (\theta_1^{(1)}, \cdots, \theta_p^{(1)})' \\
\Theta^{(2)} &= (\theta_1^{(2)}, \cdots, \theta_p^{(2)})' \\
&\vdots \\
\Theta^{(S)} &= (\theta_1^{(S)}, \cdots, \theta_p^{(S)})'
\end{align*}
$$

이 확률벡터열에서, $\Theta^{(s)}$는 오로지 $\Theta^{(s-1)}$을 통해서만 $\Theta^{(0)}, \cdots, \Theta^{(s-1)}$에 의존한다. 즉, $\Theta^{(s)}$는 주어진 $\Theta^{(s-1)}$에 대하여 $\Theta^{(0)}, \cdots, \Theta^{(s-2)}$에 조건부 독립이 된다. 즉, Markov property를 만족한다.

### Example of Gibbs Sampling

데이터가 정규분포를 따를 때의 $\mu$와 $\sigma^2$에 대해 적절한 사전분포를 이용하여 Gibbs sampling을 통한 Bayesian inference를 해보자.

예를 들어, 각 Lot별 생산된 wafer들의 수율이 정규분포를 따른다고 하자. 이 때, 50개의 Lot을 랜덤 추출했을 때의 Lot 별 수율데이터를 $X_1, \cdots , X_{n=50} \sim N(\mu, \sigma^2)$ (i.i.d.)이라고 정의하자. 

$\mu$의 사전분포로는 실수 위에서의 균등분포(improper prior), $\sigma^2$의 사전분포로는 Jeffreys' prior인 $1/\sigma^2$를 이용하자.

즉,

$$
\pi(\mu) \propto 1, \quad \pi(\sigma^2) \propto \frac{1}{\sigma^2}.
$$

> $\pi(\theta) = \pi(\mu)\pi(\sigma^2)$. 즉, 모수들 간은 independent하다고 가정.
{: .prompt-info}

그러면 데이터 $X = (X_1, \cdots , X_n)^T$에 대하여 다음과 같은 조건부 확률분포를 구할 수 있다.

$$
\mu \mid X, \sigma^2 \sim N\left( \bar{X}, \frac{\sigma^2}{n} \right)
$$

$$
\sigma^2 \mid X, \mu \sim \text{IGamma} \left( \frac{n}{2}, \frac{1}{2}\sum_{i=1}^n (X_i - \mu)^2 \right)
$$

따라서, 매 Gibbs step에서 $\mu$를 추출할 때에는 $\bar{X}$와 $\sigma^2 / n$를 평균과 분산으로 가지는 정규분포를 이용한다.

$\sigma^2$를 추출할때는 shape parameter 가 $n / 2$, scale parameter 가 $\frac{1}{2}\sum_{i=1}^n (X_i - \mu)^2$인 inverse Gamma distribution를 이용한다.

즉, Gibbs sampling algorithm은 다음과 같이 진행된다.

1. 초기값 $(\mu^{(0)}, \sigma^{2(0)})$을 정한다. 이 후, $s = 1$ 부터 $N$까지 다음 단계들을 반복한다.

2. $\mu^{(s)} \sim N\left( \bar{X}, \frac{\sigma^{2(s-1)}}{n} \right)$

3. $\sigma^{2(s)} \sim \text{IGamma} \left( \frac{n}{2}, \frac{1}{2}\sum_{i=1}^n (X_i - \mu^{(s)})^2 \right)$

## Convergence of MCMC
MCMC의 수렴여부를 나타내는 방법으로는 **Gelman and Rubin 진단**이 주로 사용된다. 이는 체인간(between-chains) 분산과 체인내(within-chain) 분산을
이용한 진단법이다.

우선 길이가 $N$인 $M>1$개의 chain이 있다고 가정하자.

이 때, 모형의 모수 $\theta$에 대하여 $$\hat{\theta}_m, \hat{\sigma}_m^2$$를 각각 $m$번째 체인으로 구한 표본 사후 평균과 표본 사후 분산이라고 하자. 또, 전체 표본 사후 평균을 $$\hat{\theta} = \frac{1}{M}\sum_{m=1}^M \hat{\theta}_m$$이라 하자.

이 경우, 체인간 분산(B)과 체인내 분산(W)을 다음과 같이 계산한다.

$$
B = \frac{N}{M-1} \sum_{m=1}^M (\hat{\theta}_m - \hat{\theta})^2, \quad W = \frac{1}{M} \sum_{m=1}^M \hat{\sigma}_m^2
$$

이후, B와 W을 이용하여 다음의 통계량을 계산한다.

$$
R = \sqrt{\frac{d + 3 \hat{V}}{d + 1} \frac{\hat{V}}{W}}
$$

이 때, $\hat{V}$(특정 정상 조건에서 합동 분산)은 다음과 같다.

$$
\hat{V} = \frac{N-1}{N} W + \frac{M+1}{MN} B, \quad d = \frac{2 \hat{V}^2}{\text{Var}(\hat{V})}
$$


위 통계량 $R$을 **Potential Scale Reduction Factor (PSRF)** 이라고 부른다.
만약 $M$개의 chain이 목표로 하는 사후 분포로 수렴하면, PSRF의 값은 1에 근접하게 된다.

따라서, 모형의 모든 모수에 대하여 $R < 1.2$를 만족하면 MCMC가 수렴한다고 말할 수 있다.