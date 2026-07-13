# Glomerulus lesion screen

ORN 전체를 seed로 두고 glomerulus를 하나씩(그리고 쌍으로) 무력화해서, 모델상 influence 분포가
얼마나 교란되는지로 glomerulus 중요도를 매긴다. **행동 예측이 아니라 모델상 분포 변화다.**

```
python -m glom_screen.check              # 불변식 검증 — 먼저 통과해야 한다
python -m glom_screen.run                # 단일 54개 × 2변종  (~40초)
python -m glom_screen.combos --variant pn  # 쌍 1,431개  (~7분)
python -m glom_screen.run_edges            # 엣지(시냅스) 수준 (~35초)  <- (6), 최종 결과
python -m glom_screen.report              # 순위 + 강건성 진단
python -m glom_screen.alpha_probe         # 아래 (1)의 근거 재현
```
실행 환경: WSL `~/miniforge3/envs/fly_neuron/bin/python` (petsc4py/slepc4py가 여기에만 있음).
**시뮬레이터 본체(`InfluenceCalculator/`)는 수정하지 않는다.**

---

## (1) 가장 중요한 것: α를 고정해야 한다

`InfluenceCalculator.calculate_influence`를 그대로 쓰면 **이 실험은 아무것도 측정하지 못한다.**

`_normalize_W`는 W의 최대 실수부 고유값이 `lambda_max`(0.99)가 되도록 `α = lambda_max / λ_max(W)`로
스케일한다. 그런데 병변을 가하면 λ_max가 떨어지므로 **α가 위로 재계산되고, 살아남은 시냅스가 전부
증폭된다.** 그리고 그 증폭량은 "병변이 얼마나 중요했는가"에 비례하기 때문에, 측정하려는 효과를
거의 정확히 상쇄한다.

`alpha_probe.py`로 54개 glomerulus에 대해 실측한 결과:

| | D_mag_sum 범위 |
|---|---|
| **α 고정** (이 스크린이 쓰는 방식) | 0.016 ~ **0.589** |
| **α 재계산** (라이브러리 기본값) | 0.000 ~ 0.023 |

- α 재계산은 **신호의 95.6%를 흡수**한다.
- 남은 잔여물은 진짜 효과와 **Spearman = −0.06** — 상관이 없다. 순위가 사실상 무작위가 된다.
- 6개 glomerulus는 D가 **음수**로 나온다(병변이 influence를 늘림).
- 가장 심한 예: `DL2d`는 진짜 D=0.589인데 α 재계산 시 0.0034로 뭉개진다.

그래서 `engine.Screen`은 `calculate_influence`를 우회하고, **온전한 W에서 α를 한 번만 구해
모든 조건에 고정 사용**한다. 그러면 control과 lesion 사이에 달라지는 것은 silence된 열의 제거뿐이다.

부수 효과로 `r_lesion ≤ r_control`이 **모든 뉴런에서 정확히** 성립한다(`check.py` [2]에서 최대 비율
1.00000000). unsigned 모드에서 `W ≥ 0`, `r = Σ(αW)ᵏ s`이므로 열 제거는 급수에서 항을 빼는 것뿐이기
때문이다. α가 움직이면 이 성질이 깨진다.

---

## (2) 두 번째로 중요한 것: `all` 변종은 쓰지 마라

`neuron.csv`의 대상을 그대로 쓰는 `all` 변종은 **포화된다.** D_mag_sum이 상위권에서 전부
0.97~0.98이라 glomerulus 간 구분이 되지 않는다.

원인은 국소 뉴런(LN)이다. LN은 뉴런당 평균 **9.1개** glomerulus에 걸쳐 있고(한 뉴런은 54개 중 51개에
등장), 그래서 어떤 glomerulus를 죽이든 공유 LN 뭉치가 함께 날아간다.

| | 뉴런당 평균 소속 glomerulus | N 민감도 (ρ vs N=500) | pn과의 일치 |
|---|---|---|---|
| `pn` (PN만, 326뉴런) | 2.6 | **0.994 ~ 0.999** | — |
| `all` (non-seed 전부, 781뉴런) | 5.5 | 0.777 ~ 0.982 | **0.210** |

두 변종의 순위 상관은 **0.21** — 사실상 불일치다. `pn`은 N 선택에 완전히 무감각하고(0.99+) 동적
범위도 건강하므로(0.016~0.589), **`pn`을 헤드라인으로 쓴다.**

---

## (3) 지표

control = ORN 전체 seed, 병변 없음.

**universe**(순위 대상)는 모든 조건에서 고정한다 — seed ORN 2,931개, 어떤 조건에서든 silence되는
뉴런 781개, control influence가 0인 뉴런을 제외한 **126,860개**.

**코호트는 control 기준으로 고정한다.** 뉴런이 병변 후 top-N 밖으로 밀려나도 명단에 남긴다
(안 그러면 가장 크게 다친 뉴런이 평균에서 빠져 낙폭을 체계적으로 과소평가한다).

```
cohort    = top_N(r_control | universe)

D_set     = 1 − |cohort ∩ top_N(r_lesion | universe)| / N        # 상위권 구성 변화
D_mag_sum = 1 − Σ_{i∈cohort} r_les,i / Σ_{i∈cohort} r_ctrl,i     # 영향력 가중 낙폭
D_mag_mean= mean_{i∈cohort} (1 − r_les,i / r_ctrl,i)             # 뉴런 동등 낙폭

Importance = (D_set + D_mag) / 2
```

셋 다 절대 [0,1]이다. **z-score를 쓰지 않는다** — 단일과 쌍을 같은 자로 비교해야 하는데, z-score는
표준화 패널이 바뀌면 값이 통째로 달라져 그 비교가 불가능해진다.

`D_mag`는 합계 가중과 뉴런 동등 가중 둘 다 계산해 저장한다(헤드라인은 결과를 보고 고를 수 있게).
실제로는 두 값이 거의 같게 나왔다.

---

## (4) 조합 (쌍 1,431개)

`pairs_pn.csv`. **단조성 위반 0건** — `D(A∪B) ≥ max(D(A), D(B))`가 1,431쌍 전부에서 성립.
파이프라인의 가장 강력한 자기검증이다.

시너지는 잔존율 `R = 1 − D_mag_sum`의 곱셈 독립 모델로 잰다:
`synergy = R(A)·R(B) − R(A∪B)`. 양수면 조합이 독립 가정보다 더 파괴적이라는 뜻.

**초가법적(synergy > 0)인 쌍은 1,431개 중 0개다.** 중앙값 −0.023, 최대 −0.0005.
즉 **모든 쌍이 예외 없이 중복적(redundant)** 이다.

단, 이걸 생물학적 발견으로 읽기 전에:

- **포화 효과가 크다.** `Spearman(synergy, D_a + D_b) = −0.788` — 병변이 강할수록 더 중복적으로
  보인다. 이미 상위권 영향력의 절반을 날렸으면 두 번째 병변이 없앨 게 남아있지 않다.
  반면 `Spearman(synergy, n_overlap) = −0.558`로 대상 중복과의 연관은 그보다 약하다.
- 다만 **대상이 전혀 겹치지 않는 720쌍도 전부(720/720) 중복적**이다. 그러니 중복성이 단순히
  같은 뉴런을 두 번 죽여서 생기는 것만은 아니다 — 경로 수렴이 실재한다.
- 곱셈 독립 모델 자체가 모델링 선택이다. 병렬 경로 제거에는 가법 모델이 더 자연스러울 수 있으나,
  가법 모델은 D가 1을 넘어버려(DL2d+VC5 = 1.08) [0,1] 경계를 존중하지 않는다.

**VM6 계열은 사실상 같은 병변이다.** VM6/VM6l/VM6m/VM6v는 PN을 9~13개씩 공유(Jaccard 0.32~0.57)
하고, D도 0.364~0.396으로 거의 같고, 조합해도 D가 **전혀 늘지 않는다**(VM6l+VM6m: 각각 0.3691,
0.3690 → 합쳐도 0.3691). 순위 8/9/10위인 이 셋은 **독립적인 발견이 아니다.**

---

## (5) ⚠ 원래 순위는 glomerulus 정체성을 재고 있지 않다

`pn` 변종의 대상조차 그 glomerulus 고유의 PN이 아니다. **다중 glomerulus PN(`M_*`)과
다른 glomerulus의 PN이 대부분이다.**

| glomerulus | 대상 PN | 그중 **자기** PN (`cell_type` 첫 토큰 == glomerulus) |
|---|---|---|
| DL2d (1위) | 31 | **9** (나머지는 `M_adPNm8`, `DL2v_adPN`×4, `VL2a_vPN`×3, `DP1l_adPN` …) |
| VC5 (3위) | 30 | **1** |
| VM6 | 18 | **1** (`VM6_adPN` 하나) |

전체 862개 pn 슬롯 중 "자기 glomerulus 소속"은 **171개뿐**이고, **12개 glomerulus는 자기 PN이
0개**다(DM1, VA6, VA2, VA7l, VM6l/m/v …). 그래서 엄격한 uniglomerular 변종은 아예 만들 수 없다.

결과:

```
Spearman(importance, 대상 PN 총수)        = +0.580
Spearman(importance, 그 glom의 자기 PN수) = +0.117     <-- 거의 무관
```

**즉 순위는 "그 glomerulus에 얼마나 많은 회로가 가지를 뻗고 있나"를 재고 있지,
"그 glomerulus 고유의 출력 경로가 얼마나 중요한가"를 재고 있지 않다.**
`"이 glomerulus가 후각 신호 전달에 중요하다"`는 문장으로 결론을 쓰면 과잉 해석이다.
현재 데이터로 정당한 문장은 `"이 glomerulus에 가지를 뻗은 회로를 제거하면 influence 분포가
이만큼 교란된다"` 정도다.

**→ 이 문제는 (6)의 엣지(시냅스) 수준 병변이 근본적으로 없앤다. 아래를 반드시 읽을 것.**

---

## (6) 엣지(시냅스) 수준 병변 — **오염 문제의 정공법. 최종 결과는 여기.**

`python -m glom_screen.run_edges`  (54×2 scope, ~35초) → `edges.csv`

(5)의 문제 — 뉴런을 죽이면 다중-glomerulus PN까지 같이 죽어서 병변이 g에 특이적이지 않다 — 는
**뉴런 대신 시냅스를 자르면 사라진다.** ORN 쪽이 모호성 없이 깨끗하기 때문이다:

> 모든 ORN은 `cell_type = ORN_<glomerulus>`로 **정확히 하나의** glomerulus에 속한다.
> 따라서 19,245개의 ORN→X 엣지가 각각 **정확히 하나의** glomerulus에 귀속되고,
> 두 glomerulus의 엣지 집합 중복은 **정확히 0**이다 — 운이 아니라 구조적으로.

다중-glomerulus PN은 **죽지 않는다.** g로부터의 입력만 잃고, 다른 glomerulus 입력과 자기 출력을
전부 유지한다. 그게 "glomerulus g가 아무것도 전달하지 않는다"의 정확한 의미다.

### `orn` scope를 쓴다. `orn_local`은 포화된다.

| scope | 자르는 것 | D_mag_sum (min/중앙/max) | 판정 |
|---|---|---|---|
| **`orn`** | ORN(g)→X 엣지 전부 | 0.0004 / 0.018 / **0.204** | **사용** |
| `orn_local` | 위 + neuron.csv[g] 내부 엣지 | 0.388 / **0.888** / 0.943 | **폐기** |

`orn_local`은 54개 중 **46개가 D > 0.80** — 구분력이 없다. glomerulus당 국소 엣지를 79~4,100개
(중앙값 1,507) 자르는데 그게 AL의 **공유 내부 배선**이라, 뭘 자르든 전부 무너진다.
`all` 뉴런 변종과 **똑같은 실패**다. "둘 다 g에 가지를 뻗었으면 g 안의 시냅스"라는 휴리스틱은
시냅스 좌표가 없어서 쓴 대용품이었고, 실패했다.

`orn`은 N에 강건하다: ρ vs N=500 이 0.930(N=50) ~ 0.997.

### 검증

`orn` scope는 **ORN(g)의 열을 0으로 만드는 것과 수학적으로 동일**해야 한다(ORN의 출력 전부 =
ORN의 열). `run_edges.py`가 실제로 두 경로를 대조하고 불일치하면 즉시 중단한다 — CSR 위치
조회가 틀렸는지 잡는 장치다. **통과했다.**

### 엣지 수준 순위 (`orn`, N=500)

| glomerulus | ORN | 엣지 | 시냅스 | D_mag_sum | importance |
|---|---|---|---|---|---|
| **DP1l** | 48 | 722 | 13,195 | 0.204 | **0.113** |
| **DP1m** | 13 | 269 | 6,207 | 0.122 | 0.067 |
| VL2p | 49 | 549 | 7,829 | 0.088 | 0.046 |
| VA2 | 85 | 588 | 8,273 | 0.084 | 0.043 |
| DM1 | 89 | 513 | 7,885 | 0.075 | 0.041 |

### ⚠ 이 순위를 읽는 법

`Spearman(importance, 시냅스 수) = +0.849`. 순위가 시냅스 수와 거의 같이 움직인다.

**크기를 보정하지 않는다.** 엣지 수준에서 크기는 **진짜 생물학**이기 때문이다 — DP1l이 ORN
시냅스 13,195개를 갖는 건 파리의 실제 특징이고, 끊으면 실제로 더 많이 망가진다.
"어느 glomerulus를 끊으면 가장 많이 망가지나"가 질문이면 답에 크기가 포함되는 게 맞다.

> **따라서 이 결과는 "DP1l이 특별한 요충지다"가 아니라
> "DP1l이 뇌로 가장 많은 후각 입력을 보낸다"로 읽어야 한다.**

### 뉴런 수준 결과는 무엇이었나

```
Spearman(엣지 'orn', 뉴런 'pn') = +0.329
```

**두 순위가 크게 어긋난다.** 뉴런 수준 1위였던 **DL2d**는 엣지 수준에서 D_mag=0.0248로
한참 아래다 — ORN이 29개뿐이라 입력을 끊어도 거의 안 움직인다. DL2d의 1위는 그 glomerulus에
가지를 뻗은 **남의 PN들**을 죽여서 나온 값이었다. 이 불일치가 (5)에서 지적한 오염의 직접 증거다.

---

## (7) 남은 한계

- **ORN은 silence되지 않는다.** `neuron.csv`의 대상 5,093건 중 **828건이 ORN = seed**이고,
  `calculate_influence`는 seed를 silence 대상에서 자동 제외한다. 사용자 결정에 따라 그대로 두었다.
  → 이건 "glomerulus 완전 무력화"가 아니라 **"감각기관은 살아있고 출력 회로만 차단"** 이다.
- **D_set은 D_mag보다 훨씬 작다** (pn에서 0.004~0.146 vs 0.016~0.589). 병변은 상위권 순위를
  뒤섞기보다 전체를 아래로 끌어내린다. 그래서 `Importance`는 사실상 `D_mag`가 지배한다.
- 40개 seed ID와 3개 대상 ID는 W에 없어 조용히 버려진다(`target_audit.csv`에 기록).
- 결과는 모델상 influence 분포 변화이지 행동 예측이 아니다.

---

## 출력

```
glom_screen_results/
  target_audit.csv   glomerulus별 listed / not_in_W / seed_excluded / effective
  singles.csv        변종 × glomerulus × N  →  D_set, D_mag_sum, D_mag_mean, importance_*
  pairs_pn.csv       쌍 1,431개 + synergy
  edges.csv          ★ 엣지(시냅스) 수준 — 귀속 모호성 0. scope=orn 을 쓸 것
  alpha_probe.csv    α 고정 vs 재계산 비교
  target_audit.csv   glomerulus별 listed / not_in_W / seed_excluded / effective
  cache/             조건별 raw influence 벡터(float32) — 지표 재정의 시 재solve 불필요
```
