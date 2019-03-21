# 복잡한 모델을 해석하는 방법, SHAP

머신러닝에서 모델의 정확도(accuracy)와 설명력(interpretability)는 흔히 트레이드 오프 관계라고 설명합니다. 하나를 선택하면 다른 하나는 희생할 수 밖에 없다는 것이죠. 예를 들어 선형회귀 모델 같이 간단한 모델을 사용하게 되면 각 데이터의 예측치가 어떻게 그 값이 나오게 되었는지는 인풋 데이터인 X에 추정한 계수 베타를 곱해주면 바로 알 수 있습니다.  하지만 선형모델은 선형 조합으로 예측할 수 없는 데이터에 대해서는 트리 앙상블 모델이나 딥러닝 모델에 비해 예측의 정확도면에서는 성능이 떨어지게 되는 경우가 많습니다. 반면 딥러닝은 높은 정확도를 보이지만 흔히 그 속을 알 수 없어 블랙박스(black-box)라고 부르죠. 우리는 정확도와 설명력 중 어떤 것을 선택해야 할까요? 둘 다 가져가는 방법은 없을까요? 

이러한 고민에서 시작되어 딥러닝과 같이 해석이 어려웠던 모델의 예측을 이해하고 설명하는 다양한 방법이 연구되었습니다. 이번 글에서는 그 대표적인 방법 중 하나인 SHAP(**SH**apley **A**dditive ex**P**lanation)를 소개하고자 합니다. 본 글의 참고자료는 최하단에 제시하였습니다. 







## Additive Feature Attribution Methods

- 간단한 모델은 모델 자체로 결과를 설명 가능
- 반면 복잡한 모델(앙상블 트리모델, 딥러닝 등)은 좀 더 심플한 ***"explanation model"***이 필요함
- explanation model이란? any interpretable approximation of the original model



#### Interpretable ML for precision medicine을 만들기 위해서는?

1. 해석 가능한 예측을 해야 함 
   - individualized explanations can engender trust on the prediction result and reveal novel rist factors
   - 각 예측에서 각각의 피쳐가 어떤 importance를 가지는지 봄
2. 모델을 트레이닝 하기 전에, 어떤 피쳐를 선택할지 고민해야 함
   - Y에 영향을 관련이 있을 것 같은(likely relevant) 피쳐를 미리 고르거나 알아내야 함
   - time series데이터는 embedding 해서 ML 모델에 넣어야 함. 의미있는(관련있는) 패턴 을 뽑아내야 함. 해석 가능한 패턴을 뽑아내야 함. 이때, 같은 피쳐지만 다른 시간에 들어온 피쳐의 중요도는 어떻게 알 수 있을까? 





#### 연구 맥락

- Hypoxemia는 발생 5분전에 예측할 수만 있어도 유의미한 조치를 취할 수 있음
- 20가지 static features와 45가지 real-time(dynamic features)를 사용하여 실시간 예측을 함![1553056623521](C:\Users\nrchu\AppData\Roaming\Typora\typora-user-images\1553056623521.png)







![1553056738060](C:\Users\nrchu\AppData\Roaming\Typora\typora-user-images\1553056738060.png)





- accuracy 와 interpretability는 트레이드 오프 관계에 있음
- **하나의 sample, 즉 한 patient에 대한 특정 prediction에 대한 feature importance를 살펴봄**

![1553056824338](C:\Users\nrchu\AppData\Roaming\Typora\typora-user-images\1553056824338.png)

![1553083404398](C:\Users\nrchu\AppData\Roaming\Typora\typora-user-images\1553083404398.png)



![1553083394533](C:\Users\nrchu\AppData\Roaming\Typora\typora-user-images\1553083394533.png)

















#### SHAP의 시각화 방법

1) Force plot

SHAP를 이해하기 위해서는 몇 가지 개념을 알아둘 필요가 있습니다. 

- base value : 모델에 트레이닝 데이터를 넣어 나온 아웃풋(y)의 평균치. 

  

이 그림은 어떤 데이터(이 코드의 경우 첫 번째 데이터)를 모델에 넣었을 때, base value로부터 해당 데이터의 아웃풋이 결정되기까지 각 피쳐가 어떻게 기여했는지 보여줍니다. 빨간 그래프 부분은 아웃풋(예측값)을 높여준 피쳐이며, 아웃풋(예측값)을 낮추는 피쳐입니다. 

이번에는 이 플롯을 전체 데이터에 대해서 그려보면 어떻게 될까요? 아래 플롯은 위에서 보여드린 플롯을 90도 돌려서 가로로 쭉 쌓은 그래프입니다. 







**References**

- 논문 : 
  - [Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765-4774)](https://arxiv.org/abs/1705.07874)

- 유튜브(논문 저자 Su-in Lee의 강의):
  - [Su-In Lee (UW): Interpretable Machine Learning in Precision Medicine](https://www.youtube.com/watch?v=M2bD7Dt9MxI)
- Python 모듈: 
  - [SHAP 파이썬 모듈](https://github.com/slundberg/shap) 
- 블로그 글 
  - [Interpretable Neural Networks](https://towardsdatascience.com/interpretable-neural-networks-45ac8aa91411), [Gabriel Tseng](https://medium.com/@gabrieltseng)
  - [Demystifying Black-Box Models with SHAP Value Analysis](https://medium.com/civis-analytics/demystifying-black-box-models-with-shap-value-analysis-3e20b536fc80), [Gabriel Tseng](https://medium.com/@gabrieltseng)
  - (번역) [해석가능한 XGBoost 기계학습](https://medium.com/@aldente0630/%ED%95%B4%EC%84%9D%EA%B0%80%EB%8A%A5%ED%95%9C-xgboost-%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-26621610adb5), [aldente0630](https://medium.com/@aldente0630)

### 









- Global level vs. Local level

  전통적인 머신러닝 모델의 featrue importance는 모델 전반에 걸쳐 어떤 변수가 가장 중요한지 보여줍니다. 예를 들어 트리모델에서는 트리에서 가지를 쳐 내려가는 데 어떤 변수가 가장 많이 사용되었는지가 feature importance를 뜻하게 되는 것이죠. 선형회귀모델에서는 베타값의 절대값이 클수록 y값 변동에 큰 영향을 미치는 중요 변수가 될 것입니다. 

  하지만 데이터 전반에 걸친 변수의 중요도나 영향력이 개별 데이터에 대해서도 공통적이진 않을 것입니다. 예를 들어 