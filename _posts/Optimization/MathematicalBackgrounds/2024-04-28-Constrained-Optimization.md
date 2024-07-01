---
title: Constrained Optimization
author: rdh
date: 2024-04-28T02:24:33.530Z
categories: [Optimization, Mathematical Backgrounds]
tags: [optimization]
math: true
---

## Constrained Optimization
### General Constrained Optimization

$$
\begin{aligned}
& \text{minimize}
& & f(x) \\
& \text{subject to}
& & g_i(x) \le 0, \quad i=1,\dots,l\\
& & & h_j(x) = 0, \quad j=1,\dots,m\\
\end{aligned}$$

* $f$ : **objective function**.
* $g_i$ : **inequality constraints**.
* $h_j$ : **equality constraints**.
* 제약 조건을 만족하는 $x$의 집합을 **feasible set** 이라고 한다.

* 점 $x$에 대해 $A(x)=\{i:g_i(x)=0\} \cup \{j:h_j(x)=0\}$을 **active set** 이라고 한다.
* 주어진 점 $$x^*$$과 active set $$A(x^*)$$에 대해, $$\{\nabla g_i(x^*), i \in A(x^*)\}$$이 linearly independent이면, **LICQ** (Linearly Independence Constraint Qualification)가 성립한다고 말한다.

> LICQ 성립 시, active constraint의 gradient는 0이 될 수 없다.
{: .prompt-tip}

* The problem can be rewrite as:

$$
\begin{aligned}
& \text{minimize}
& & f(x) \\
& \text{subject to}
& & \mathbf{g}(x) \le 0\\
& & & \mathbf{h}(x) = 0\\
\end{aligned}$$

* 주어진 문제의 **Lagrangian** 은 다음과 같이 정의된다.

$$
L(x,\nu) = f(x) + \sum_{i=1}^l \lambda_i g_i(x) + \sum_{j=1}^m \nu_j h_j(x) = f(x) + \lambda^T \mathbf{g}(x) + \nu^T \mathbf{h}(x)$$

## Karush-Kuhn-Tucker (KKT) Conditions
### Motivation for the KKT Conditions
하나의 inequality constraint를 가진 optimization problem을 고려하자.

$$
\begin{aligned}
& \text{minimize}
& & f(x) \\
& \text{subject to}
& & g(x) \le 0\\
\end{aligned}$$

이는 다음과 같이 표현될 수 있다.

$$
\begin{aligned}
 \text{minimize} \quad f_{\infty-step}(x) &:= \begin{cases} f(x) &\text{if } g(x)\le 0 \\ 0 &\text{if } g(x)> 0 \end{cases} \\
&= f(x) + \infty \cdot 1_{g(x)>0}
\end{aligned}
$$

이 때, $f_{\infty-step} = \max\limits_{\lambda \ge 0}$ $L(x,\lambda)$ where $L(x,\lambda)= f(x) + \lambda g(x)$.

![](/assets/img/constrained-optimization-01.png){: width=300}

따라서, 해당 optimization problem은 다음과 같이 새롭게 표현될 수 있다:

$$
\min\limits_{x}\max\limits_{\lambda \ge 0}L(x,\lambda) := f(x)+\lambda g(x)$$


### KKT Conditions

**Theorem: First-Order Necessary Conditions (KKT conditions)**
> Suppose that $$x^*$$ is a local solution of constrained optimization problem and the LICQ holds at $$x^*$$. 
> 
> $$
\begin{aligned}
& \text{minimize}
& & f(x) \\
& \text{subject to}
& & g_i(x) \le 0, \quad i=1,\dots,l\\
& & & h_j(x) = 0, \quad j=1,\dots,m\\
\end{aligned}
>$$
>
> Then there is a Lagrangian multiplier vector $$(\lambda^*, \nu^*)$$, such that the following KKT conditions are satisfied at $$(x^*, \lambda^*, \nu^*)$$:
> 
>$$
\begin{aligned}
&\text{(1)} \quad \nabla_x L(x^*,\lambda^*,\nu^*) = \nabla_x f(x^*) + \sum_{i=1}^l\lambda_i^*\nabla_x g_i(x^*) + \sum_{j=1}^m\nu_j^*\nabla_x h_j(x^*) = 0 \\
&\text{(2)} \quad g_i(x^*) \le 0, \forall i \\
&\text{(3)} \quad h_j(x^*) = 0, \forall j \\
&\text{(4)} \quad \lambda_i^* \ge 0, \forall i \\
&\text{(5)} \quad \lambda_i^* g_i(x^*) = 0, \forall i
\end{aligned}
>$$

각 조건은 다음과 같이 불린다.
* (1): Stationarity 조건
* (2), (3): Primal feasibility 조건
* (4): Dual feasibility 조건
* (5): Complementary slackness 조건

## Lagrangian Duality
### Primal and Dual Problem
* **Primal problem**

$$
\begin{aligned}
& \text{minimize}
& & f(x) \\
& \text{subject to}
& & g_i(x) \le 0, \quad i=1,\dots,l\\
& & & h_j(x) = 0, \quad j=1,\dots,m\\
\end{aligned}$$

<p align=center>or</p>

$$
\min\limits_{x}\max\limits_{\lambda \ge 0, \nu}L(x,\lambda,\nu) := f(x)+\lambda^T g(x)+\nu^T h(x)$$

* **Lagrangian Dual problem**

$$
\max\limits_{\lambda \ge 0, \nu}D(\lambda,\nu):=\min\limits_{x}L(x,\lambda,\nu)$$

### Duality Theorem
#### Weak Duality
Dual problem의 optimal value는 primal problem의 optimal value의 lower bound이다.

**Theorem: Weak Duality**
> Let $x$ and $(\lambda,\nu)$ be a feasible solution to primal and dual problems, respectively.
> Then $f(x) \ge D(\lambda,\nu)$.
> Moreover, if $$f(x^*)=D(\lambda^*,\nu^*)$$, then $$x^*$$ and $$(\lambda^*,\nu^*)$$ solves the primal and dual problems, respectively.

* Duality gap: $$f(x^*)-D(\lambda^*,\nu^*)$$

#### Strong Duality
Convex optimization problem의 경우, dual problem과 primal problem의 optimal value는 같다.

**Theorem: Strong Duality**
> Let $f,g_i$ be convex and $h_j$ be affine for all $i,j$. Then $$f(x^*) = D(\lambda^*,\nu^*)$$ if the optimal value is finite.

* **Affine functions** $f(x_1,...,x_n)=A_1x_1+...+A_nx_n+b$ 형태의 function. 

#### Wolfe Duality
Strong Duality를 이용하면 주어진 optimization 문제를 다르게 표현할 수 있다.

**Theorem: Wolfe Duality**
> Let $f,g_i$ be convex, $h_j$ be affine for all $i,j$, and $$x^*$$ be an optimal solution of the primal.
> Then $$(x^*,\lambda^*,\nu^*)$$ at which LICQ holds solves the Wolfe dual problem of the form
> 
>$$
\begin{aligned}
& \text{minimize}
& & L(x,\lambda,\nu) \\
& \text{subject to}
& & \nabla_xL(x,\lambda,\nu)=0\\
& & & \lambda \ge 0\\
\end{aligned}$$

> $\nabla_xL(x,\lambda,\nu)=0$는 $L(x,\lambda,\nu)$의 local optima에 대한 조건이다.
{: .prompt-info}

### Linear and Quadratic Programming
#### Linear Programming (LP)

$$
\begin{aligned}
& \text{minimize}
& & c^Tx \\
& \text{subject to}
& & Ax=b\\
& & & x \ge 0\\
\end{aligned}$$

* Lagrangian dual $L(x,\lambda,\nu) = c^Tx-\lambda^Tx+\nu^T(b-Ax)$은 convex이다.
* 따라서, $c-\lambda-A^T\nu=0$ and $\lambda\ge 0$ 일 때, minimum을 얻는다.
	* Check that $c-\lambda-A^T\nu=0$ and $\lambda\ge 0$ $\Longleftrightarrow$ $A^T\nu \le c$.
* 그 결과, dual problem은 다음과 같다.

$$
\begin{aligned}
& \text{maximize}
& & b^T\nu \\
& \text{subject to}
& & A^T\nu \le c\\
\end{aligned}$$

#### Quadratic Programming (QP)

$$
\begin{aligned}
& \text{minimize}
& & \frac{1}{2}x^THx+d^Tx \\
& \text{subject to}
& & Ax \le b\\
\end{aligned}$$

where $H$ is symmetric and positive semidefinite. (따라서, objective function은 convex이다.)

* Lagrangian dual $L(x,\lambda) = \frac{1}{2}x^THx+d^Tx+\lambda^T(Ax-b)$은 convex이다.
* 따라서, $Hx+A^T\lambda+d=0$ or $x=-H^{-1}(d+A^T\lambda)$ 일 때, minimum을 얻는다.
* 그 결과, dual problem은 다음과 같다.

$$
\begin{aligned}
& \text{maximize}
& & \frac{1}{2}\lambda^TD\lambda+\lambda^Tc-\frac{1}{2}d^TH^{-1}d \\
& \text{subject to}
& & \lambda \ge 0\\
\end{aligned}$$

where $D=-AH^{-1}A$ and $c=-b-AH^{-1}d$.

> 이 dual problem은 단순히 nonnegative domain에서의 concave quadratic function maximization 문제이기에, 상대적으로 쉽게 풀 수 있다.
{: .prompt-tip}