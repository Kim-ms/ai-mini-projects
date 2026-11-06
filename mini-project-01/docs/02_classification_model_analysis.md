# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** 검증 데이터 셋
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367
- **LogisticRegression의 Validation 혼동행렬:** TN 1,526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
기준 모델(Dummy model)은 다수 클래스를 선택하기에, 미구매자가 다수(약 84%)인 불균형 데이터에서 정확도는 84%로 높게 나오지만, 소수인 구매자에 대한 예측에서 모두 오답이기에 Recall과 Precision가 0이 나오고, 두 지표의 조화 평균인 F1-Score 또한 0이 됨.
___

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 / 0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
학습 초기 Training Loss의 급감 후 완만하게 하향/상승 안정화되는 것으로 보아 모델 학습이 안정적임을 알 수 있음. 또한, Train·Valid F1 점수가 동반 상승하고 두 곡선 간의 간격이 일정하게 유지되는 것으로 보아, 모델이 과적합되지 않고 양호한 일반화 성능을 유지하며 학습됨. 다만 최종 검증의 F1 Score(0.633)으로 보아 모델의 일반화 성능의 뛰어나다고 보기엔 어려움이 있음.
추가로, epoch의 진행에 따른 loss 감소율과, F1 Score의 상승율로 미루어보아 추가 학습으로 모델 성능의 향상은 기대하기 어려우며, 추가 학습 시 오히려 과적합 문제를 야기 할 수 있음.
___

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** MLP-A(0.633) / MLP-B(0.645)로 MLP-B가 근소한 차이(0.012)로 더 높음.
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:**
  로지스틱 회귀 대비 MLP-A는 Precision에서 유사한 성능(0.734 / 0.717)을 보이나, Recall에서 기존 0.367에서 0.566으로 향상(0.199)되었고, MLP-B는 Precision에서 기존 0.734에서 0.575로 감소하였으나, Precision 감소량(0.157)량 대비 Recall에서 기존 0.367에서 0.734로 대폭 향상(0.367)되었음
- **Train·Validation F1 차이에서 관찰한 점:** Train F1 0.692 / Valid F1 0.645으로 학습·검증간 간극은 0.047로 매우 안정적임. 이전의 안정적인 결과를 보였던 MLP-A(0.077)와 비교해보아도 더욱 안정적이며 일반화 성능이 뛰어남.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.
(지표는 Valid 기준, A > B 순) Precision : 0.717 > 0.575 // Recall : 0.566 > 0.734 // F1-Score : 0.633 >> 0.645
소수 클래스에 대한 양성 가중치를 보다 적영히야 소수 클래스 오답에 대한 패널티를 강화하여(구매자를 구매자로 예측 할 수 있도록) Recall에서 유의미한 상승(+0.168)이 나타났으나, 그에 대한 반동으로 Precision이 감소(-0.142)하였음.
F1-Score의 경우 0.012로 소폭 증가하여 유의미 한 차이는 아니라고 판단되지만 프로젝트 목표인 구매자 예측에는 보다 적합해보임.
___

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** Precision 하강 / Recall 상승 / FP 상승 / FN 저하
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** Precision 상승 / Recall 하강 / FP 하강 / FN 상승
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B(weighted BCE), Threshold : 0.5, F1-Score : 0.645
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:**
임계값이 0.3에서 0.7로 커질수록 두 모델 모두 Precision은 상승하고 Recall은 하락하는 Trade-off가 관찰됨.
MLP-A에서는 임계값 0.3 지점에서 F1-Score가 가장 높고, 임계값이 커질 수록 Recall과 Precision의 차이가 확대되는 경향이 있음. 양성 가중치를 적용한 MLP-B는 임계값의 증가에 따라 Recall, Precision, F1-Score의 간극이 축소되었고, 0.7 구간에서 지표가 가깝게 나타남, 다만 0.5 지점에서 종합 성능인 F1-Score가 높게 평가 되어, Recall과 Precision의 균형을 고려한 종합 성능이 가장 뛰어났으며, 구매 여부 예측이 중요한 본 프로젝트에서는 F1-Score가 가장 높고 Recall 성능도 우수한 0.5 모델이 가장 우수하다고 판단함.


**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다.  
___

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B (weighted BCE) / Threshold : 0.5
- **최종 재학습 데이터(Train + Validation):** 10,480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 적용
- **Test Accuracy / Precision / Recall / F1:** 0.858 / 0.526 / 0.843 / 0.648
- **Test 혼동행렬:** TN 1,347 / FP 217 / FN 45 / TP 241
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 저장 전후 예측 일치: True
- **`predict_purchase()` 실행 결과:**
예측: {'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.5}
실제 Revenue: 0

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
___

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** 양성 클래스 가중치를 적용한 MLP-B(th=0.5)가 F1-Score 0.645, Recall 0.734로 가장 우수한 종합 분류 성능을 기록하였으며, 로지스틱 회귀 대비 구매자 탐지 성능이 크게 향상됨.
- **남아 있는 모델의 한계:** F1-Score 0.645만으로 충분한 일반화 성능을 확보했다고 판단하기 어려우며, Precision 감소에 따른 FP 증가와 모델의 표현력 한계가 남아 있음.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
