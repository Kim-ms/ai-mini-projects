# 002. 탐지 성능 실험 및 오류 분석

**프로젝트:** YOLO 객체 탐지 성능 평가 및 최적화 (CV)  
**작성 범위:** CP-02~CP-10

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. 각 `___`를 실제 값 또는 1~2문장으로 바꾸세요. 상세한 선택 이유와 트레이드오프는 `ADR.md`에 작성합니다.

## 1. 모델 및 평가기 구현 (CP-02~CP-06)

- **사용 모델:** YOLO26 Nano (`yolo26n.pt`), 사전학습 가중치 사용
- **기본 입력 이미지 크기(`imgsz`):** 640
- **평가용 IoU Threshold:** 0.5
- **기준 Confidence Threshold:** 0.30
- **평가 지표:** TP, FP, FN, Precision, Recall, F1
- **예측–정답 매칭에서 TP로 판정하는 조건:** Confidence Threshold가 0.3, IoU가 0.5 이상이며 예측이 정답 클래스와 일치하여, 하나의 정답과 한 예측이 매칭된 경우 TP로 판정.
- **여러 이미지의 TP·FP·FN 집계 방법:** 모든 이미지의 TP·FP·FN을 합산한 뒤 **데이터셋 전체 기준** Precision·Recall·F1을 산출
- **IoU·매칭·집계·성능 지표 검증 셀 실행 결과:** CP-06 평가 지표 단위 테스트 통과.

## 2. Confidence Threshold별 성능 비교 (CP-07)

`threshold_df`의 Validation 전체 성능을 기록합니다. Confidence 0.30은 CP-06의 기준 설정입니다.

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.15 | 440 | 183 | 265 | 0.706 | 0.624 | 0.663 |
| 0.30 | 379 | 53 | 326 | 0.877 | 0.538 | 0.667 |
| 0.60 | 234 | 8 | 471 | 0.967 | 0.332 | 0.494 |

- **Validation F1이 가장 높은 Confidence:** 0.30 (f1 : 0.667)

**분석:** Confidence를 높였을 때 FP·FN과 Precision·Recall이 실제로 어떻게 변했으며, 그 이유는 무엇인지 작성합니다.  
___

> 참고 결과: `outputs/tables/cp07_threshold_comparison.csv`, `outputs/figures/cp07_threshold_metrics.png`

## 3. 클래스별 성능 및 탐지 오류 분석 (CP-08)

기준 Confidence 0.30에서 `class_error_df`의 결과를 기록합니다. 정답 객체 수는 TP+FN입니다.

| 클래스 | 정답 객체 수 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Person | 383 | 215 | 24 | 168 | 0.900 | 0.561 | 0.691 |
| Car | 204 | 111 | 19 | 93 | 0.854 | 0.544 | 0.665 |
| Bicycle | 75 | 22 | 6 | 53 | 0.786 | 0.293 | 0.427 |
| Bus | 43 | 31 | 4 | 12 | 0.886 | 0.721 | 0.795 |

- **FN이 가장 많은 클래스 / Recall이 가장 낮은 클래스(정답이 있는 클래스 기준):** Person / Bicycle

### 대표 오류 이미지 3건

`error_images`에서 선정한 이미지의 정답과 예측을 비교합니다. 원인은 객체 크기·가림·복잡한 배경 등 실제 이미지에서 관찰한 근거를 바탕으로 추정합니다.

| 이미지 ID | FP / FN | 관찰한 탐지 오류 및 원인 후보 |
|---|---|---|
| 000000254814 | 4 / 20 | | **다수 소형 객체 밀집·겹침 미탐(FN) 및 배경 오탐(FP):** GT 24개(person 13, bicycle 8, car 2, bus 1) 중 교차로에 밀집한 소형 이륜차·보행자 객체들이 소형 및 밀집·겹침으로 미탐지됨. 반면 GT 라벨이 빠진 상단·우측 하단 승합차 2대, 도로 중앙의 구조물와 버스를 승합차로 오인을 오탐지하여 FP 4건 발생.(TP=4, FP=4, FN=20)|
| 000000460147 | 4 / 10 | **원거리 소형·적재 차량 미탐(FN) 및 비대상(라벨 외) 차량 오탐(FP):** GT 18개(car 13, bus 2, person 3) 중 정상적으로 탐지한 차량 8건 외 원거리의 소형 객체 및 겹친 차량·버스 등 10개가 미탐지됨. GT에 미포함되어있는  자동차 3건과 카캐리어 후면부를 탐지하여 FP 4건 발생. (TP=8, FP=4, FN=10)|
| 000000378139 | 0 / 14 | **원거리 소형 객체 및 복잡한 배경 속 객체 전량 미탐(FN):** 전체 GT 14개 중 12개(bicycle 6, car 5, person 1)가 강 건너편 다리 난간과 건물 배경에 밀집·중첩된 아주 작은 크기로 존재하여 모델이 단 1건도 탐지하지 못했고, 유람선 승객 중 GT에 포함되어 있는 2개의 객체(person 2) 또한 탐지하지 못함(TP=0, FP=0, FN=14). |

**가설 검증:** `01_cv_data_analysis.md`의 3절에서 세운 가설이 실제 결과와 일치했는지, 클래스별 정답 수·Small 비율·Recall·대표 오류를 근거로 작성합니다.  

가설 : 경계가 뚜렷하지 않은 객체와 원거리의 소형 객체에 대하여 미탐지되는 경우가 발생할 수 있으며, 명확하지 않은 Ground Truth로 인해 정상적으로 예측한 것을 오탐지로 처리될 수 있음. 또한, 표본의 수 적은 클래스의 경우 적은 오탐 및 미탐지에도 성능 지표가 민감하게 영향을 받게 되어 지표가 일반화 성능을 대변 한다고 보기엔 어려움이 있음.

실제 결과 : 겹침, 소형 객체에 대해 탐지에 있어 미탐지가 다수 발생하는 결과를 보였으며, 오탐의 경우 실제 이미지 및 GT 검토 오탐은 매우 소수였음.

___

> 참고 결과: `outputs/tables/cp08_class_error_analysis.csv`, `outputs/figures/cp08_error_<image_id>.png`

## 4. 최종 장면 분석기 확인 (CP-09)

- **`analyze_scene()`에 입력한 이미지 ID 또는 파일명:** ___
- **탐지된 클래스별 객체 수:** ___
- **탐지 목록(`detections`), 클래스별 개수(`counts`), 결과 이미지(`rendered_path`) 반환 확인:** ___

> 활용 목적·우선 평가 지표·최종 Confidence의 선택 근거는 `ADR.md`의 CV 의사결정에 작성합니다.

## 5. 최종 Validation·Test 성능 (CP-10)

- **Test 평가 전에 확정한 Confidence(`SELECTED_CONF`):** ___
- **평가 이미지 수:** Validation ___장 / 보류 Test ___장

동일한 Confidence에서 `final_df`의 결과를 기록합니다.

| 데이터 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | ___ | ___ | ___ | ___ | ___ | ___ |
| 보류 Test | ___ | ___ | ___ | ___ | ___ | ___ |

**해석:** Validation과 Test의 Precision·Recall·F1 차이를 실제 수치로 설명하고, Test의 클래스별 정답 객체 수(`test_class_df`의 TP+FN)와 데이터 규모를 고려해 차이의 원인을 추정합니다.  
___

**평가의 한계:** 이 소규모 데이터로 실제 교통 환경의 탐지 성능을 보장할 수 없는 이유와 추가로 확인할 점을 작성합니다.  
___

> 참고 결과: `outputs/tables/cp10_final_validation_vs_test.csv`, `outputs/tables/cp10_test_per_class.csv`, `outputs/reports/cp10_final_summary.json`
>
> Test 결과를 확인한 뒤 Confidence를 다시 선택하지 않습니다.
