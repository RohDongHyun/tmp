---
title: "Data Preprocessing Process"
author: rdh
date: 2024-03-02 14:14:25.962
categories: [03. Machine Learning, 02. Techniques for ML]
tags: [data preprocessing, machine learning]
math: true
---

## Data Cleaning
**Data cleaning**이란 데이터의 품질을 높이기 위해 다양한 규칙을 검사하여 데이터의 일관성을 유지하고 오류를 방지하는 작업이다. Data preprocessing에서 필수적이고 중요한 단계이다.

일반적으로 data cleaning을 위한 방법으로는 **data discrepancy detection**이 있다. 이는 metadata(예. data 범위, dependency 등)을 이용하여 data의 discrepancy(불일치)를 감지하고 수정하는 것을 의미한다. Data discrepancy detection의 종류로는 다음과 같은 것들이 있다.

* **Uniqueness rule**: uniqueness 제약이 존재하는 attribute에서 중복된 값이 있는지 확인

* **Consecutive rule**: 연속적인 값 (예. 연도, 일련번호 등)을 갖는 attribute에서 누락된 값이 없는지 확인

* **Null rule**: null 조건을 나타내는 방식 (예. 빈 칸, 특수 문자 등)이 일관성있게 적용되어 있는지 확인하고, 그러한 null 값들에 대한 처리가 제대로 되어있는지 확인

* **Data scrubbing**: 간단한 domain 지식을 기반으로 오류를 감지하고 수정

* **Data auditing**: 데이터 간의 규칙과 관계를 분석하여 이를 위반하는 데이터를 감지하고 수정

이 때, 가장 많이 요구되는 기술은 missing data와 outlier를 포함한 noisy data에 대한 처리 기술이다. 해당 내용은 현재까지도 다양한 기법들이 등장하고 있기에, 여기서는 간단한 개념과 방식에 대해서만 설명한다.

### Missing Data
Missing data를 적절히 처리하는 것은 무척 중요한 일이나, 또한 굉장히 어려운 일이기도 하다. 현재까지도 어떠한 방법이 우위에 있다고 말할 수 없기에 계속해서 연구가 되어야 하는 부분이다.

* **Ignore the tuple** 

주로 missing data는 해당 data를 사용하지 않는 것으로 해결하는 경우가 많다. 이는 전체 dataset의 크기에 비해 missing data의 size가 작은 경우에는 효과적이다. 하지만, 현실의 많은 문제에서는 missing data의 비율이 높거나, 또는 특정 조건의 값이 일괄적으로 누락되는 경우 (예. 특정 온도 이상에서는 signal 수집 실패 등)에는 이러한 방법이 잘못된 분석 결과를 야기할 수 있다.

* **Fill in it automatically**

또 다른 방법으로는 특정한 값으로 일괄 채워버리는 것이 있다.
이 때, '특정한 값'을 설정하는 방법은 다양하다. 특정한 constant 값으로 채울 수도 있고, attribute의 mean 또는 median과 같은 statistic을 사용하기도 한다. Labeled data의 경우, 동일 label을 갖는 data를 이용해서 채우기도 한다.

* **Inference-based method**

통계적 모형을 이용하여 missing data를 예측하여 이를 채우는 방법도 있다. Missing data가 없는 instance들을 이용한 regression model로 예측하는 방법과, attribute의 분포를 가정하여 sampling하는 방법 등이 있다.

* **Weak supervision**

Machine learning 과정에서 자주 문제가 되는 missing data는 바로 label이 없는 data이다. 일반적으로 label이 없는 데이터는 기존에 잘 알려진 supervised learning을 바로 적용할 수 없기에 label을 따로 추가하는 작업이 필요하며, 이는 곧 시간과 금전적인 비용의 증가를 의미한다. 이러한 이유로, label이 없는 데이터를 활용한 training 방법이 많이 연구되어 왔다.

### Noisy Data
데이터 noise는 데이터 수집 도구의 결함, 데이터 입력 시의 실수, 데이터 전송 과정에서의 오류, 기술적 제한 등의 이유로 발생할 수 있다.

이러한 noise는 데이터 중복, 누락, 모순 등을 유발할 수 있기 때문에 적절한 처리가 필요하다. 특히 outlier data를 적절하게 처리하지 못한다면 결과의 큰 왜곡을 발생할 수 있으므로 특히 주의해야한다.

* **Binning**

데이터를 정렬하고, 구간화한 뒤에 각 구간 별 mean 또는 median 등을 활용하여 data를 smoothing한다.

* **Clustering**

Hierarchical clustering 등의 방법을 이용해서 이상치를 감지하고 제거한다.

* **Smoothing**

Regression model등을 이용하여 data의 smoothness를 증가시킨다.

* **Combined computer and human inspection**

알고리즘을 통해 의심스러운 값을 감지하고, 인간이 해당 값을 검토하고 처리한다. 일반적인 3-sigma outlier detection 등이 여기에 해당한다.

* **Data visualization**

저차원 data의 경우 data visualization을 통해 한 눈에 noise와 outlier를 알아낼 수 있다. 

## Data Integration
**Data integration**은 여러 data source로부터 추출한 data를 통합하여 하나의 저장소에 저장하는 과정을 말한다. 이를 통해 하나의 database나 data warehouse를 이용한 데이터 분석 및 학습이 가능하다.

* **Schema integration**

Data integration 시에는 우선 서로 다른 data source의 schema를 일치시키는 작업이 필요하다. 서로 다른 database의 경우 column 명칭이 다르거나, 데이터 type이 다른 경우가 빈번하므로 이러한 부분을 통일시키는 작업이 필요하다.

* **Entity identification problem**

Entity identification problem이란 여러 data source에서 동일한 entity를 밝혀내는 문제를 말한다. 예를 들어, "John Smith"와 "Jonathan Smith"은 각각 동일한 고객을 나타낸다고 볼 수 있기 때문에 이를 식별하는 작업이 필요하다.

* **Detecting and resolving data value conflicts**

서로 다른 data source의 경우, 동일한 entity의 attribute 값이 서로 다른 경우가 있다. 이러한 충돌을 해결하기 위해 데이터 분석을 통해 data redundancy 또는 discrepancy 등을 식별하고 일관성 있는 값으로 통합해야 한다.

이 때, 가장 흔하게 발생하는 문제는 data redundancy이다.

### Data Redundancy

일반적으로 data integration 과정의 가장 큰 이슈는 동일한 정보가 여러 번 나타나는 redundancy이다. 이러한 redundancy를 해결하지 않으면 data의 분포가 왜곡되어 분석 결과가 달라질 수 있기 때문에 꼭 처리해야하는 이슈이다. Data redundancy는 다음과 같은 것들이 존재한다.

* Record level redundancy: 하나 이상의 record가 완전히 동일한 데이터를 포함하는 경우

* Attribute level redundancy: 동일한 데이터베이스나 테이블에서 동일한 정보가 여러 attribute에 중복되어 있는 경우

* Dataset level redundancy: 두 개 이상의 데이터 집합에 동일한 정보가 중복되어 있는 경우

이러한 data redundancy를 찾아내기 위한 방법으로 **correlation analysis** 또는 **covariance analysis**를 수행한다.

## Data Reduction
**Data reduction**이란, data의 차원 또는 양을 줄이는 과정으로, 원본 데이터의 특성을 보존하면서 불필요한 정보를 제거하여 데이터를 간결하고 효율적으로 만드는 과정을 말한다. 이는 machine learning 및 데이터 분석에서 중요한 단계 중 하나로 모델의 성능을 향상하고 계산 비용을 절감하는 역할을 한다. 

### Dimensionality Reduction
Dimensionality reduction 이란, 데이터의 차원을 줄이는 방법으로, 불필요한 attribute를 제거하거나 attribute의 표현 방법을 변형하여 데이터를 간소화한다.

> Machine learning에서 curse of dimensionality를 방지하기 위한 기법이다.
{: .prompt-info}

* **Feature selection**

Feature selection이란 실제 분석 또는 학습에 필요한 attribute만을 선택하는 작업을 말한다. Attribute의 모든 부분 집합을 검토하거나, p-value와 같은 statistic를 활용하여 불필요한 attribute를 제거해나가는 방식이 사용된다. 사용자가 자신의 domain knowledge를 활용한 방식도 가능하다.

* **Principal Component Analysis (PCA)**

PCA는 가장 널리 사용되는 dimensionality reduction 기법으로, 원본 데이터의 분산을 가장 잘 표현하는 projection axis들을 찾아 data를 표현하는 방법이다. 이를 통해 데이터의 차원을 줄이면서도 원래 데이터의 대부분 정보를 보존할 수 있다.

* **Wavelet Transform**

Wavelet transform은 시간 및 주파수 영역의 특징을 동시에 분석하는 수학적 기법이다. 데이터의 고주파 및 저주파 성분을 분리하여 정보를 보존하면서도 차원을 줄일 수 있다. 특히 이미지 처리 및 신호 처리와 같은 분야에서 효과적으로 사용된다.

### Numerosity Reduction
Numerosity reduction은 data point의 수를 줄이는 전략으로, 데이터를 집약하여 표현한다. 특히, 대용량 데이터를 효율적으로 처리해야 하는 bigdata 분석 및 real-time 처리 시스템과 같은 상황에서 유용하다.

* **Parametric methods**

Parametric method는 regression과 같은 수학적 모델을 활용하여 데이터를 더 간결한 형태로 표현하는 것을 말한다. 전체 dataset을 저장하는 대신 모델의 parameter만 저장하면 된다.

* **Nonarametric methods**
Nonarametric method는 특정한 수학적 모델에 의존하는 대신, histogram, clustering, sampling과 같은 기법을 사용하여 데이터를 압축하는 것을 말한다. 이러한 방법은 데이터의 분포 또는 관계를 간결하게 요약하는 데 초점을 둔다.

### Data Compression
Data compression이란 데이터를 효율적으로 저장하기 위해 데이터를 압축하는 전략이다. 원본 데이터의 압축된 표현을 얻어 저장 공간을 절약하고 처리 효율성을 높일 수 있다. 

* **Lossless Compression**

원본 데이터를 재구성할 때, 정보 손실이 없는 압축 방법을 lossless compression이라고 한다. Run-Length Encoding (RLE), Huffman Coding 및 Lempel-Ziv-Welch (LZW)와 같은 기술이 일반적으로 사용된다.

* **Lossy Compression**

더 높은 압축률을 위해 데이터 정확도의 일부를 희생하는 압축 방법이다. JPEG image에서의 discrete cosine transform (DCT) 또는 JPEG 2000에서의 discrete wavelet transform (DWT)과 같은 기술이 사용될 수 있다.

### Data Sampling
Data sampling이란 대규모 데이터 집합에서 작은 sample을 추출하는 과정으로, 전체 데이터를 대표할 수 있는 작은 집합을 얻는 것을 말한다. 이는 여러 data mining 알고리즘을 전체 데이터에 적용하는 것보다 훨씬 효율적인 방법이다.

* **Simple random sampling**

모집단의 각 항목을 동일한 확률로 선택하는 sampling 방법이다. 간단하고 이해하기 쉽지만, 데이터의 특성을 고려하지 않고 무작위로 샘플을 추출하기 때문에 특정 패턴이나 구조를 잡아내기 어려울 수 있다.

* **Sampling without replacement**

한 번 선택된 항목은 모집단에서 제거하는 sampling 방법이다. 따라서 동일한 항목이 두 번 이상 선택되지 않습니다. 이 방법은 각 항목의 중복 선택을 방지하고, 다양한 데이터를 대표하는 샘플을 얻을 수 있습니다.

* **Sampling with replacement**

선택된 항목을 모집단에 다시 포함하는 sampling 방법이다. 즉, 동일한 항목이 여러 번 선택될 수 있다. 이 방법은 모집단의 크기가 충분히 크거나 중복 선택이 허용되는 경우 유용하다.

* **Stratified sampling**

데이터를 여러 부분집합으로 분할하고, 각 부분집합에서 동일한 비율로 샘플을 추출하는 방법이다. 이 방법은 모집단의 특성을 고려하여 샘플을 추출하기 때문에 전체 모집단을 더 잘 대표할 수 있다다. 특히 데이터가 unbalance 한 경우에 유용하며, 각 부분집합에서 동일한 비율로 샘플을 추출함으로써 bias를 줄일 수 있다.

## Data Transformation

Data transformation은 데이터의 표현을 분석과 학습에 용이하도록 변형하는 것을 말한다. Data transformation을 어떻게 하느냐가 분석의 성패를 좌우하는 경우가 많으므로, data preprocessing에서 가장 중요한 단계로 볼 수 있다. 

* **Smoothing**

데이터에서 노이즈를 제거하여 데이터의 불규칙성을 줄이는 과정으로, 주로 moving average 또는 filtering method 등을 사용한다.

* **Feature construction**

기존의 attribute를 기반으로 새로운 attribute를 만드는 과정이다. 이를 통해 데이터의 의미를 확장하거나 더 유용한 특징을 도출할 수 있다.

* **Normalization**

데이터를 특정 범위 내로 scaling하여 데이터의 분포를 조정하는 과정이다. 주로 min-max normalization 또는 z-score normalization과 같은 기법이 사용된다. 이는 데이터의 단위나 크기를 일관되게 처리하여 모델의 성능을 향상시키고 데이터를 더 쉽게 비교할 수 있도록 만든다.

* **Discretization**

Continuouse attribute를 구간으로 변환하는 과정을 말한다. 이를 통해 데이터의 noise를 무시하거나, 산포에 덜 민감하도록 만들 수 있다.

이 중, 가장 많이 사용되는 기법은 normalization과 discretization이다.

### Normalization

* **Min-max normalization**

Attribute $A$에 대한 min-max normalization은 다음과 같다.

$$
v' = \frac{v-\min A}{\max A - \min A}
$$

이를 통해, $A$의 범위가 $[\max A, \min A]$에서 $[0, 1]$로 조정된다.

* **Z-score normalization**
Attribute $A$의 mean과 standard deviation이 각각 $\mu_A, \sigma_A$라고 할 때, $A$에 대한 z-score normalization은 다음과 같다 (standard normalization이라고도 한다). 

$$
v' = \frac{v-\mu}{\sigma_A}
$$

변환된 데이터는 평균을 중심으로 대칭적으로 분포하게 된다. 즉, 각 값들이 평균에서 얼마나 벗어나는 지를 동일한 기준으로 나타낸다.

### Discretization

* **Binning**: 각 구간을 동일한 폭 또는 동일한 개수의 data를 갖도록 나눈다.
  * Equal-width partitioning: attribute를 동일한 크기의 구간으로 나누는 방법이다. 직관적이지만 outlier가 큰 영향을 미칠 수 있고 unbalanced data에 대해서 잘 처리되지 않을 수 있다.
  * Equal-depth partitioning: 특정 개수의 구간으로 나누되, 각 구간에 대략 동일한 개수의 sample이 들어가도록 나누는 방법이다.

* **Clustering analysis**: 데이터를 서로 유사한 그룹으로 묶어 구간을 형성한다. Hierarchical clustering, K-means clustering 기법 등을 사용한다.

* **Decision-tree analysis**: Decision-tree를 사용하여 데이터를 이산화하는 방법