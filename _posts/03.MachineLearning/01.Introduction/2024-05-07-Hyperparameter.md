---
title: Hyperparameter
author: rdh
date: 2024-05-07T07:48:05.299Z
categories: [03. Machine Learning, 01. Introduction to Machine Learning]
tags: [hyperparameter, machine learning]
math: true
---
## Hyperparameter
**Hyperparameter**는 일반적인 model의 parameter와 다르게, 학습 과정으로부터 배우는 parameter가 아니라 학습 과정을 control하기 위해 사용자가 직접 설정하는 parameter를 말한다. 일반적인 model의 parameter는 learnable parameter라고도 한다.

> Hyperparameter는 dataset로부터 최적 값을 계산할 수 없지만, dataset에 따라서 최적의 값이 달라진다.
{: .prompt-info}

## Hyperparameter Tuning
Hyperparameter는 model (특히 deep model)의 performance에 중요한 영향을 미치므로, 적절한 값을 찾아내기 위한 노력이 중요하다. 이 과정을 **hyperparameter tuning** 또는 hyperparameter optimizing이라고 하며, 일반적인 hyperparameter tuning의 목적은 test error를 낮게 만드는 값을 찾아내는 것이다. 

따라서, 최적의 hyperparameter는 CV 등을 적용하여 validation set error를 최소로 만드는 값으로 선택하는 것이 일반적이다.

![](/assets/img/hyperparameter-02.png){: width="650"}

### Hyperparameter Tuning Methods

기본적으로 hyperparameter는 empirical하게 최선의 조합을 찾아야 한다. 이 때 주로 두가지 방법으로 hyperparameter tuning이 진행된다.

* **Grid Search**
    * 각 hyperparameter에 대해서 가능한 값들에 대한 후보군을 설정하고, 모든 hyperparameter 조합에 대해서 측정해보는 것이다.
    * 일반적으로, 후보군을 설정할 때 logarithmic scale로 설정한다 (예. {50, 100, 250, 500, 1000}).
    * Computational cost가 exponential하게 증가할 수 있다.

* **Random Search**
    * Hyperparameter 값에 대한 distribution을 설정하고, random sample을 통해 최선의 조합을 찾아내는 것이다.
    * Logarithmic하게 sampling하기 위해, log-scaling된 hyperparameter 값에 대해서 uniform하게 sampling 한다 (예. log_learning_rate ~ uniform(-1,-5)).
    * 상대적으로 grid search보다 더 빠르게 좋은 조합을 찾을 수 있다.

![](/assets/img/hyperparameter-01.png){: width="650"}

