---
title: Nonparametric Statistics
author: rdh
date: 2024-02-04 11:33:00 +0800
categories: [Statistics, Introduction to Statistics]
tags: [nonparametric statistics, robust methods, statistics]
math: true
---
Parameter estimation의 경우 기본적으로 모집단의 분포를 가정(정규분포 등)한 후, 해당 분포의 parameter를 추정하고 검정하였다 ([참고](https://rohdonghyun.github.io/posts/Parameter-Estimation-and-Hypothesis-Test/)). 하지만, 모집단에 대해 정규분포 같은 구체적인 분포함수를 가정하는 것이 무리일 때에는 모집단 분포에 대한 가정을 약화시켜 오류의 가능성을 줄이는 **비모수 (Nonparametric)** 방법을 고려할 수 있다.

## Nonparametric Hypothesis Test
비모수적 방법에서는 모집단의 분포에 대한 가정을 최대한 약화시킨다. 따라서, 분포의 연속성과 경우에 따라 대칭성 만을 가정한다.

일반적으로 비모수 추론에 사용되는 값들은 관측값의 **부호 (sign)** 또는 **순위 (rank)**에 기초한 점수 (score)이다. 즉, 관측값 자체를 사용하지 않고, 모집단에 분포에 의존하지 않는 값들을 사용한다.

### One Sample Sign Test
다음과 같은 가정 하에서 위치 모수 $\mu$에 대한 검정을 진행하려고 한다.

* 데이터: 모집단으로부터 크기 $n$인 확률표본의 관측값 $X_1, \cdots , X_n$

* 가정
    1. 기본 모형: $X_i = \mu + e_i$ ($i=1,\dots, n$). 여기서 $\mu$는 미지의 위치 모수, $e$는 오차항
    2. $n$개의 오차항 $e$들은 i.i.d.
    3. 오차항 $e$는 0에 대해서 대칭인 분포를 따름

위치 모수 $\mu$에 대한 검정은 일반적으로 세 가지 가설에 대해 진행한다:

1. $H_0: \mu = \mu_0$ vs. $H_1: \mu > \mu_0$
2. $H_0: \mu = \mu_0$ vs. $H_1: \mu < \mu_0$
3. $H_0: \mu = \mu_0$ vs. $H_1: \mu \ne \mu_0$

위 문제에 대한 비모수 검정은 **부호검정 (Sign test)**를 사용한다. 부호검정은 위치모수에 대한 비모수적 검정 중 가장 오래되고 간단한 검정법으로, 귀무가설 $H_0$하에서 위치모수의 값 $\mu_0$보다 큰 관측값의 개수만을 이용하여 검정을 진행한다.

부호검정통계량은 다음과 같다.

$$
B = \sum_{i=1}^n I(X_i-\mu_0)
$$

$I(x)$는 $x>0$이면 1, 아니면 0인 값을 갖는 indicator 함수이다. 즉, $B$는 $\mu_0$보다 큰 관측값의 개수를 나타낸다.

위 부호검정 통계량을 이용한 부호검정은 다음과 같이 진행한다.

1. $H_1: \mu > \mu_0$ 일 때, $B \geq b(\alpha, n)$이면 $H_0$을 기각
2. $H_1: \mu < \mu_0$ 일 때, $B < b(1-\alpha, n)$이면 $H_0$을 기각
3. $H_1: \mu \ne \mu_0$ 일 때, $B \geq b(\alpha/2, n)$이거나 $B < b(1-\alpha/2, n)$이면 $H_0$을 기각

여기서 $b(\alpha, n)$은 표본의 크기가 n일 때 $H_0$ 하에서 부호검정통계량 $B$의 상위 $100\alpha$ 백분위수로 $P_0[B\geq b(\alpha,n)] = \alpha$을 만족하는 값이다.

#### Estimation based on Sign Test

위치모수 $\mu$에 대한 추정은 다음과 같은 절차로 부호검정에 기초하여 진행할 수 있다.

1. 관측값을 크기순으로 배열: $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$
2. 부호검정통계량을 이용한 $\mu$의 추정량 $\hat{\mu}$은 관측값들의 중앙값으로 설정:

$$
\hat{\mu} = \begin{cases} 
X_{(k+1)}, & n = 2k + 1 \text{ 일 때} \\ 
\frac{X_{(k)} + X_{(k+1)}}{2}, & n = 2k \text{ 일 때}
\end{cases}
$$

이러한 추정량 $\hat{\mu}$은 윌콕슨 부호검정에 기초한 $\mu$의 핫지스-레만 일표본추정량 (Hodges-Lehmann one-sample estimator)이라 한다.

> 일반적으로 검정의 power와 estimation의 accuracy는 비례하는데, 부호검정의 power는 낮은 편이다. 따라서, 부호검정으로 추정한 median 값은 대개 부정확하다.
{: .prompt-info}

### Wilcoxon Signed-Rank Test
일표본(one-sample) 위치모수에 대해 가장 널리 사용되는 비모수적 검정법으로 **윌콕슨 부호순위검정 (Wilcoxon Signed-Rank Test)**이 있다. 여기서는 단순히 관측값이 $\mu_0$보다 크거나 작다는 것만 고려하는 것이 아니라 관측값의 상대적인 크기도 함께 고려하여 검정을 진행한다.

부호검정에서는 오차항의 대칭성의 가정이 필요없지만 부호순위검정에서는 분포의 대칭성 가정이 필요하다.

1. 모든 $i = 1, \ldots, n$에 대해, $Z_i = X_i - \mu_0$를 계산한다.
2. $\vert Z_1\vert, \vert Z_2\vert, \ldots, \vert Z_n\vert$ 중에서 $\vert Z_i\vert$의 순위를 $R_i^+$라고 한다.
3. 윌콕슨 부호순위검정통계량: $W^+ = \sum_{i=1}^n \psi(X_i - \mu_0) \cdot R_i^+$
4. 검정법: 유의수준 $\alpha$에서
    * $H_1: \mu > \mu_0$일 때, $W^+ \geq w^+(\alpha, n)$이면 $H_0$ 기각
    * $H_1: \mu < \mu_0$일 때, $W^+ \leq w^+(1 - \alpha, n)$이면 $H_0$ 기각
    * $H_1: \mu \neq \mu_0$일 때, $W^+ \geq w^+(\alpha/2, n)$ 또는 $W^+ < w^+(1 - \alpha/2, n)$이면 $H_0$ 기각

여기서 $w^+(\alpha, n)$은 $H_0$하에서 부호순위검정통계량 $W^+$의 상위 $100\alpha$ 백분위수를 나타낸다.

> 위치 모수에 대한 추정량으로는 실제적으로 $t$-검정 기반 추정량과 윌콕슨 부호 순위 검정 기반 추정량만을 주로 사용한다.
{: .prompt-tip}

#### Estimation based on Signed-Rank Test
오차항 $e$ 분포의 대칭성을 가정할 수 있는 경우에 위치모수 $\mu$에 대한 점추정은 윌콕슨 부호순위 검정통계량을 이용하여 다음과 같이 구할 수 있다.

1. $N = n(n+1)/2$개의 모든 $i \leq j (i, j = 1, 2, \ldots, n)$에 대해:
 
    $$
    W_{ij} = \frac{X_i + X_j}{2}
    $$

    을 계산한다. 이때 $W_{ij}$을 월쉬평균(Walsh average)이라 한다.

2. 월쉬평균 $W_{ij}$의 순서통계량을 $W_{(1)}, W_{(2)}, \ldots, W_{(N)}$이라 하면, $\mu$의 점추정량 $\hat{\mu}$은 $W_{ij}$의 중앙값으로 정의된다:

    $$
    \hat{\mu} = 
    \begin{cases}
    W_{(k+1)}, & \text{if } N = 2k + 1 \text{일 때} \\
    \frac{W_{(k)} + W_{(k+1)}}{2}, & \text{if } N = 2k \text{일 때}
    \end{cases}
    $$

이러한 추정량 $\hat{\mu}$은 윌콕슨 부호순위검정에 기초한 $\mu$의 핫지스-레만 일표본추정량 (Hodges-Lehmann one-sample estimator)이라 한다.

> Walsh average는 계산량이 많아 실제로는 잘 쓰이지 않는다.
{: .prompt-warning}

### Two-Sample Wilcoxon Rank Sum Test
두 모집단에서 얻어진 확률표본으로부터 각 모집단의 위치모수에 대한 추정과 검정문제를 이표본 위치문제라고 한다.

두 확률표본은 독립이며, 각 표본은 대조(control) 모집단과 처리(treatment) 모집단으로 생각할 수 있다. 이표본 위치문제는 두 모집단 간 위치모수가 차이가 있는지, 차이가 있다면 어느 정도 차이가 있는지 알아보는 것이 목적이다.

* 데이터: 두 모집단으로부터 각각 크기가 $m$과 $n$인 확률표본을 $(X_1, X_2, \ldots, X_m)$, $(Y_1, Y_2, \ldots, Y_n)$ 이라 하자. $N = m + n$으로 표기하고, 편의상 $m \geq n$이라 가정한다.

* 가정
    1. $\mu$은 $X_i$에 대한 미지의 위치모수, $\Delta$는 두 집단간의 위치모수의 차(이동모수), $e$은 오차항
    $$
    \begin{aligned}
    X_i &= \mu + e_i, \quad i = 1, \ldots, m \\
    Y_j &= \mu + \Delta + e_{m+j}, \quad j = 1, \ldots, n
    \end{aligned}
    $$

    2. $N$개의 오차항들은 서로 독립이고, 두 표본 내에서 모두 동일한 연속분포를 따른다.

* 이표본 위치문제에서 관심 있는 모수는 이동모수(shift parameter) $\Delta$이고 $\Delta$의 추정량은 $\hat{\Delta}$로 나타내고, $\Delta$에 대한 검정은 일반적으로 세 가지 가설에 대해 진행한다:
    1. $H_0: \Delta = 0$ vs. $H_1: \Delta > 0$
    2. $H_0: \Delta = 0$ vs. $H_1: \Delta < 0$
    3. $H_0: \Delta = 0$ vs. $H_1: \Delta \neq 0$

위 문제에 대해서 가장 널리 사용되는 방법으로는 **윌콕슨 순위합 검정 (Wilcoxon Rank Sum Test)**이 있다. 이는 혼합표본에서 각 관측값의 순위를 이용하는 방법으로, 다음과 같이 진행된다.

1. $X = (X_1, \ldots, X_m)$와 $Y = (Y_1, \ldots, Y_n)$의 혼합표본에서 $Y_j$의 순위를 $R_j$라고 한다.
2. 윌콕슨 순위합통계량: $$W = \sum_{j=1}^n R_j$$
3. 검정법: 유의수준 $\alpha$에서
    * $H_1: \Delta > 0$일 때, $W \geq w(\alpha, m, n)$이면 $H_0$ 기각
    * $H_1: \Delta < 0$일 때, $W < w(1 - \alpha, m, n)$이면 $H_0$ 기각
    * $H_1: \Delta \neq 0$일 때, $W \geq w(\alpha/2, m, n)$이면 또는 $W < w(1 - \alpha/2, m, n)$이면 $H_0$ 기각
    
여기서 $w(\alpha, m, n)$은 $H_0$ 하에서 순위합통계량 $W$의 분포의 상위 $100\alpha$ 백분위수를 나타낸다.

#### Estimation based on Rank Sum Test
이동모수 $\Delta$은 윌콕슨 순위합 검정통계량을 이용하여 다음과 같이 추정할 수 있다.

1. 모든 $i, j (i = 1, \ldots, m, j = 1, \ldots, n)$에 대해 $mn$개의 $V_{ij} = Y_j - X_i$를 구한다.
2. $V_{ij}$의 순서통계량을 $V_{(1)}, V_{(2)}, \ldots, V_{(mn)}$이라 하면 $\Delta$의 점추정량 $\Delta$는 $V_{ij}$의 중앙값으로 정의된다.

$$
\Delta = 
\begin{cases} 
V_{(k+1)}, & mn = 2k + 1일 때 \\
\frac{V_{(k)} + V_{(k+1)}}{2}, & mn = 2k일 때
\end{cases}
$$

### Two-Sample Test for Scale Parameters
이표본 척도문제란 두 표본의 척도모수에 차이가 있는 지를 검정하는 문제이다.

* 데이터: 두 모집단으로부터 각각 크기가 $m$과 $n$인 확률표본을 $(X_1, X_2, \ldots, X_m)$, $(Y_1, Y_2, \ldots, Y_n)$이라 하자. 이전과 동일하게 $N = m + n$으로 표기하고, 편리상 $m \geq n$이라 가정한다.

* 가정
    1. $\mu_X, \mu_Y$은 $X$와 $Y$ 표본의 위치모수, $\sigma_X, \sigma_Y$은 $X$와 $Y$ 표본의 척도모수, $e$은 오차항이다.
    
    $$
    \begin{aligned}
    X_i &= \mu_X + \sigma_X e_i, \quad i = 1, \ldots, m \\
    Y_j &= \mu_Y + \sigma_Y e_{m+j}, \quad j = 1, \ldots, n
    \end{aligned}
    $$

    2. $N$개의 오차항들은 서로 독립이고, 두 표본 내에서 모두 동일한 연속분포를 따른다.

이표본 척도문제에서는 두 척도모수 간의 비율 확인하는 방법을 사용한다. 두 척도모수 간의 비는 $\gamma = \sigma_Y / \sigma_X$로 나타내며, $\gamma$에 대한 검증은 일반적으로 세 가지 가설에 대해 진행한다:

1. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 > 1$
2. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 < 1$
3. $H_0 : \gamma^2 = 1$ vs. $H_1 : \gamma^2 \ne 1$

여기서는 두 위치모수에 대한 정보가 있는 경우, 즉 $\mu_Y - \mu_X$가 알려졌다고 가정한다. $\mu_Y - \mu_X$가 알려진 경우에는 $Y$ 관측값을 $Y$에서 두 위치모수의 차이인 $\mu_Y - \mu_X$의 값으로 대체하면 두 관측값의 위치모수는 같은 것으로 생각할 수 있기 때문에, 편의상 $\mu_Y - \mu_X = 0$이라 가정할 수 있다.

이렇게 두 모집단의 위치모수가 같은 경우에는 척도모수가 혼합표본의 순위에 절대적인 영향을 주게 되고, 이 점을 이용한 대표적인 검정은 **앤서리-브래들리 검정(Ansari-Bradley test)**이 있다.

앤서리-브래들리 검정의 절차는 다음과 같다.

1. $N = m + n$개의 관측값을 작은 값부터 크기 순서대로 나열한다.

2. 혼합표본에서 관측값 $X_i$의 순위를 $S_i$라 하면 $S_i$들은 순위 $1, 2, \ldots, N$ 중 $m$개를 차지하고, 이 $S_i$에 대해 앤서리-브래들리 스코어 $a_{AB}(S_i)$는 다음과 같이 정의된다.


    | $S_i$         | 1   | 2   | ... | $\frac{N+1}{2}$ | ... | $N-1$ | $N$ (홀수) |
    | ------------- | --- | --- | --- | --------------- | --- | ----- | ---------- |
    | $a_{AB}(S_i)$ | 1   | 2   | ... | $\frac{N+1}{2}$ | ... | 2     | 1          |

    | $S_i$         | 1   | 2   | ... | $\frac{N}{2}$ | $\frac{N}{2} + 1$ | ... | $N-1$ | $N$ (짝수) |
    | ------------- | --- | --- | --- | ------------- | ----------------- | --- | ----- | ---------- |
    | $a_{AB}(S_i)$ | 1   | 2   | ... | $\frac{N}{2}$ | $\frac{N}{2}$     | ... | 2     | 1          |


3. 앤서리-브래들리 통계량은 다음과 같이 정의한다: $T_{AB} = \sum_{i=1}^m a_{AB}(S_i)$.

4. 검정법: 유의수준 $\alpha$에서 $(\gamma^2 = \sigma_Y^2/\sigma_X^2)$,
    * $H_1 : \gamma^2 > 1$일 때, $T_{AB} \geq t_{AB}(\alpha, m, n)$이면 $H_0$ 기각
    * $H_1 : \gamma^2 < 1$일 때, $T_{AB} < t_{AB}(1 - \alpha, m, n)$이면 $H_0$ 기각
    * $H_1 : \gamma^2 \neq 1$일 때, $T_{AB} \geq t_{AB}(\alpha/2, m, n)$이거나 $T_{AB} < t_{AB}(1 - \alpha/2, m, n)$이면 $H_0$ 기각

혼합표본에서 $\gamma^2 > 1$ $(\sigma_Y^2 > \sigma_X^2)$라는 것은 $Y$들이 $X$보다 넓게 퍼져있다는 것을 의미한다. 따라서 $X_i$의 순위 $S_i$는 중앙에 가깝게 분포되면서, $S_i$에 대응되는 $a_{AB}(S_i)$은 상대적으로 큰 값을 가지게 되고, 앤서리-브래들리 통계량 $T_{AB}$가 커지게 된다.

$T_{AB}$는 $X$들이 갖는 앤서리-브래들리 스코어 합으로 정의했지만, $Y$들이 갖는 앤서리-브래들리 스코어의 합을 고려해도 비슷한 검정을 진행할 수 있다.

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
