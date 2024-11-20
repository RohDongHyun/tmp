---
title: Probability Distribution
author: rdh
date: 2024-02-01 11:33:00 +0800
categories: [01. Statistics, 01. Introduction to Statistics]
tags: [probability distribution, statistics]
math: true
---

## Probability
**Probability** refers to a measure of the likelihood of an event occurring, and the probability of event $A$ is denoted as $P(A)$.

* $P(A)$ = "Number of elements in event $A$" / "Total number of elements in the sample space"

Here, event $A$ is a subset of the sample space $S$, which represents the set of all possible outcomes resulting from an experiment $E$.

### Properties of Probability

* $P(\varnothing) = 0$
* $A \subseteq B \implies P(A) \leq P(B)$
* $0 \leq P(A) \leq 1$
* $P(A^c) = 1 - P(A)$
* $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### Conditional Probability
The **conditional probability** of event $B$ given that $A$ has occurred is denoted as $P(B\vert A)$ and, under the assumption that $P(A) > 0$, is defined as:

$$
P(B\vert A) = \frac{P(A \cap B)}{P(A)}.
$$

In other words, the conditional probability represents the likelihood of event $B$ occurring, considering event $A$ as a reduced sample space.

#### Law of Total Probability
Let us consider a partition $\\{A_1, \ldots, A_n\\}$ of the sample space $S$. A partition of the sample space satisfies the following conditions:

$$
\forall A_i \cap A_j = \emptyset \ (i \neq j), \ A_1 \cup A_2 \cup \cdots \cup A_n = S.
$$

Under this condition, the **law of total probability** is expressed as follows:

$$
P(B) = P(B \vert A_1)P(A_1) + \cdots + P(B \vert A_n)P(A_n).
$$

### Independence
If the occurrence of event $A$ has no effect on the probability of event $B$, the two events $A$ and $B$ are said to be **independent**.

When $A$ and $B$ are independent, $P(B \vert A) = P(B)$ or $P(A \cup B) = P(A)P(B)$ hold.

If $A$ and $B$ are not independent, they are referred to as **dependent**.

* If $A \cup B = \emptyset$, the events $A$ and $B$ are **mutually disjoint**, meaning they cannot occur simultaneously, and $A$ and $B$ are dependent events.
* If $A$ and $B$ are independent, $A^C$ and B$, as well as $A$ and $B^C$ are also independent.

### Bayes Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

### Random Variable
A **random variable** is a function that maps each element of the sample space to a real number.

$$
c \in S, \quad X(c)=x \in \mathbb{R}
$$

The probability that a random variable $X$ takes a value in $\mathcal{B}$:

$$
P(X \in \mathcal{B}) = P(c \in S \mid X(c) \in \mathcal{B})
$$

### Probability Distribution
The **probability distribution** of a random variable $X$ represents the possible values $X$ can take and the corresponding probabilities. It provides the information necessary to calculate the probabilities associated with $X$.

#### Discrete Random Variable
A **discrete random variable** is one where $X$ can take on discrete values such as $x_1, x_2, x_3, \dots$, providing the corresponding probabilities for each value.

The probability distribution of $X$ is represented by the **probability mass function (PMF)**, denoted as $p(x)$, which is defined as follows:

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
A **continuous random variable** is one where $X$ can take an uncountably infinite number of values, providing information to calculate the probability of $X$ falling within a specific interval.

The probability distribution is expressed using the **probability density function (PDF)**, $f(x)$, and the probability that $X$ lies within $a \leq X \leq b$ is given by:

$$
P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx
$$

* $f(x) \geq 0$
* $\int_{-\infty}^{\infty} f(x) \, dx = 1$
* For a continuous random variable, the probability at a single point is zero: $P(X=a)=0$

#### Cumulative Distribution Function (CDF)
The **cumulative distribution function (CDF)** is another function used to represent a probability distribution, in addition to the PMF or PDF. It is defined as follows (applicable to both discrete and continuous random variables):

$$
F_X(x) = P(X \leq x)
$$

* $F_X(x)$ is a non-decreasing function.
* For a continuous random variable: $\frac{d}{dx}F(x)=f(x)$

### Expectation
The **expectation** of a random variable $X$ represents its central value and is also referred to as the **mean**.

$$
\mu = \mathbb{E}(X) = 
\begin{cases} 
\sum_{x} x \, p(x) & \text{(discrete)} \\
\int_{-\infty}^{\infty} x \, f(x) \, dx & \text{(continuous)}
\end{cases}
$$

* $\mathbb{E}(X)$: 1st moment, information about the center
* $\mathbb{E}(X^2)$: 2nd moment, information about dispersion
* $\mathbb{E}((X - \mu)^2)$: 2nd centered moment
* $\mathbb{E}(X^3)$: 3rd moment, information about symmetric (**skewness**)
* $\mathbb{E}(X^4)$: 4th moment, information about the tail (**kurtosis**)

The expectation of a function $g(X)$ of a random variable $X$ is given by:

$$
\mu = \mathbb{E}(g(X)) = 
\begin{cases} 
\sum_{x} g(x) \, p(x) & \text{(discrete)} \\
\int_{-\infty}^{\infty} g(x) \, f(x) \, dx & \text{(continuous)}
\end{cases}
$$

The expectation is linear.
* $\mathbb{E}(aX+b) = a\mathbb{E}(X)+b$
* $\mathbb{E}(ag(X)+bh(X)) = a\mathbb{E}(g(X))+b\mathbb{E}(h(X))$

### Variance and Standard Deviation
Let $\mu$ be the mean of $X$. 

* **Variance**

    $$
    \text{Var}(X) = \mathbb{E}((X - \mu)^2) = 
        \begin{cases}
        \sum_x (x - \mu)^2 p(x) & (\text{discrete}) \\
        \int_{-\infty}^{\infty} (x - \mu)^2 f(x) \, dx & (\text{continuous})
        \end{cases}
    $$

* **Standard deviation**

    $$
    \text{sd}(X) = \sqrt{\text{Var}(X)}
    $$

* $\text{Var}(X) = \mathbb{E}(X^2) - [\mathbb{E}(X)]^2$
* $\text{Var}(aX + b) = a^2 \text{Var}(X)$

## Examples of Probability Distribution
### Bernoulli Distribution
A **Bernoulli trial** is an experiment where the outcome is one of two possibilities. That is, the sample space is $S = \\{\text{success}(s), \text{failure}(f)\\}$, and the success probability is $p=P(\\{s\\})$.

A **Bernoulli random variable** is a random variable that maps the outcome of a Bernoulli trial to the values 0 or 1, such that $X(s)=1$ and $X(f)=0$.

The probability distribution of a Bernoulli random variable is called the **Bernoulli distribution**, denoted as $X \sim Ber(p)$.

* $p(x) = p^x (1 - p)^{1-x}, \quad x = 0, 1$
* $\mathbb{E}(X) = p$
* $\text{Var}(X) = \mathbb{E}(X^2) - [\mathbb{E}(X)]^2 = p(1 - p)$

### Binomial Distribution
The binomial distribution represents the distribution of the number of successes in $n$ independent Bernoulli trials, each with a success probability of $p$. It is denoted as $X \sim B(n,p)$ or $Bin(n,p)$

* $p(x) = \binom{n}{x} p^x (1-p)^{n-x}, \quad x = 0, \ldots, n$
* If $n=1$, $X$ is the Bernoulli distribution.
* $\mathbb{E}(X) = np$
* $\text{Var}(X) = np(1 - p)$

### Poisson Distribution
The **Poisson distribution** models the number of independent events occurring within a fixed period of time or a specified space. It is denoted as $X \sim Poi(\lambda)$, where $\lambda$ represents the average rate of occurrence.

* $p(x) = \frac{\lambda^x}{x!} e^{-x}, \quad x = 0, \ldots, \lambda>0$
* $\mathbb{E}(X) = \lambda$
* $\text{Var}(X) = \lambda$
* When $X_1 \sim Poi(\lambda_1)$ and $X_2 \sim Poi(\lambda_2)$, $X_1 + X_2 \sim Poi(\lambda_1 + \lambda_2)$.

### Uniform Distribution
When a random variable $X$ is equally likely to take any value between $a$ and $b$, it is said to follow a **uniform distribution**, denoted as $X \sim Uniform(a,b)$.

* $f(x) = \frac{1}{b-a}, \quad a < x < b$
* $\mathbb{E}(X) = \frac{a+b}{2}$
* $\text{Var}(X) = \frac{(b-a)^2}{12}$

### Beta Distribution
The **beta distribution** is a type of continuous probability distribution for a random variable $X$ where $0 \leq X \leq 1$, and it is defined by the following PDF:

$$
f(x) = f(x \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}, \quad x \in [0,1], \, \alpha > 0, \beta > 0.
$$

It is denoted as $X \sim Beta(\alpha, \beta)$.

* $B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}$ is called the normalizing constant.
  * $\Gamma(x)$: gamma function
* $\mathbb{E}(X) = \frac{\alpha}{\alpha + \beta}$
* When $\alpha = \beta = 1$, the beta distribution becomes equivalent to the uniform distribution.

### Exponential Distribution
The **exponential distribution** models the waiting time until the next independent event occurs after a previous event. It is denoted as $X \sim Exp(\lambda)$, where $\lambda$ is the rate parameter.

* $f(x) = \lambda \exp(-\lambda x), \quad x>0$
* $f(x) = \frac{1}{\rho} \exp(- x / \rho), \quad x>0$
  * $\rho$: scale parameter
* $\mathbb{E}(X) = \frac{1}{\lambda} = \rho$
* $\text{Var}(X) = \frac{1}{\lambda^2} = \rho^2$
* Memoryless Property: $P(X>s+t \mid X>s) = P(X>t), \, s,t>0$

### Normal Distribution
The **normal distribution**, also known as the **Gaussian distribution**, was introduced by Carl Friedrich Gauss (1777–1855).

> It is a continuous probability distribution discovered during the study of probability distributions for errors in physical experiments. In the early stages of statistical development, it was believed that data histograms that did not resemble the shape of a Gaussian distribution were abnormal, leading to the term "normal."
{: .prompt-info}

It is denoted as $X \sim N(\mu, \sigma^2)$, where $\mu$ is the mean and $\sigma^2$ is the variance.

The probability density function is given as follows:

$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^{2}}}exp(-\frac{(x-\mu)^{2}}{2\sigma^{2}}), \quad -\infty<x<\infty, \, \sigma>0
$$

* $\tau = 1/\sigma^2$: precision
* $aX+b \sim N(a\mu+b, a^2\sigma^2)$ when $X \sim N(\mu, \sigma^2)$

#### Standard Normal Distribution
A **standard normal distribution** is a normal distribution with a mean of 0 and a standard deviation of 1. It is typically denoted by $Z$.

* Standardization: $Z = \frac{X-\mu}{\sigma} \sim N(0,1)$ when $X \sim N(\mu, \sigma^2)$