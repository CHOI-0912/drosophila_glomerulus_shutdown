# Glomerulus lesion screen (BANC)

초파리 커넥톰(BANC)에서 후각 수용 뉴런(ORN) 전체를 seed로 두고, glomerulus를 하나씩
무력화했을 때 모델상 influence 분포가 얼마나 교란되는지 측정한다.

**결과의 의미는 행동 예측이 아니라 모델상 influence 분포의 변화다.**

```bash
bash setup.sh                      # upstream 라이브러리 클론 + 패치 + 의존성 확인
python -m glom_screen.check        # 불변식 검증 — 먼저 통과해야 함
python -m glom_screen.run_edges    # ★ 최종 결과 (~35초)
python -m glom_screen.report       # 순위 + 강건성 진단
```

## 문서

- **[`report/glomerulus-screen.html`](report/glomerulus-screen.html)** — 결과 보고서.
  용어 정의식, 방법론적 함정 2개, 54개 전체 순위, 지표 간 상관행렬.
  브라우저로 바로 열면 된다 (외부 의존성 없는 단일 파일).
- **[`glom_screen/README.md`](glom_screen/README.md)** — 방법·지표·한계, 근거 수치 포함.
- **[`CHANGES.md`](CHANGES.md)** — 라이브러리 가드 패치의 근거.

---

## 이 저장소에 없는 것

- **시뮬레이터** (`InfluenceCalculator`) — [DrugowitschLab/ConnectomeInfluenceCalculator][up]에서
  `setup.sh`가 클론한다. 커밋 `57cc08c`로 고정. 남의 코드를 복사해 들고 다니지 않는다.
- **BANC 데이터** (`banc_888_*.feather`, 347MB) — GitHub 100MB 파일 제한 초과.
  저장소 루트에 직접 놓아야 한다. `setup.sh`가 존재 여부를 검사한다.

[up]: https://github.com/DrugowitschLab/ConnectomeInfluenceCalculator

## 이 저장소에 있는 것

| | |
|---|---|
| `glom_screen/` | 스크린 본체 (10개 모듈 + README) |
| `orn.txt` | seed ORN 2,971개 |
| `neuron.csv` | glomerulus별 병변 대상 (54행) |
| `make_silences.py` | `neuron.csv`를 생성한 스크립트 |
| `glom_screen_results/*.csv` | 실행 결과 (`edges.csv`가 최종) |
| `library-guards.patch` | 라이브러리의 조용한 실패 2개를 막는 패치 |
| `CHANGES.md` | 그 패치의 근거 |
| `a.py` | 초기 탐색용 스크립트 (라벨 오류 있음, 참고용) |

## 핵심 결과 세 가지

**1. α를 고정해야 한다.** 라이브러리 기본값은 병변된 W에서 α를 다시 계산하는데,
병변이 중요할수록 λ_max가 많이 떨어져 α가 크게 보정되고, **그게 측정하려는 효과를 정확히
상쇄한다.** 실측: 측정값 D의 **95.6%가 증발**하고, 남은 4.4%는 진짜 D와 **상관이 −0.06**.
순위가 무작위가 된다. → `glom_screen/engine.py`가 온전한 W에서 α를 한 번만 구해 고정.
근거 재현: `python -m glom_screen.alpha_probe`

**2. 뉴런이 아니라 시냅스를 잘라야 한다.** `neuron.csv`의 PN 대상 862슬롯 중 그 glomerulus에
실제로 속하는 PN은 **171개뿐**이고, 12개 glomerulus는 자기 PN이 **0개**다. 뉴런을 죽이면
남의 glomerulus 회로까지 같이 죽는다. 반면 ORN은 `cell_type = ORN_<glom>`으로 **정확히 하나의**
glomerulus에만 속하므로, ORN 시냅스를 자르면 **귀속 모호성이 0**이다.

**3. 순위는 시냅스 개수와 함께 움직인다** (ρ = +0.849). 따라서 결과는
*"이 glomerulus가 특별한 요충지다"*가 아니라 *"이 glomerulus가 뇌로 가장 많은 후각 입력을
보낸다"*로 읽어야 한다.

## 검증

`python -m glom_screen.check` — 전부 수학적으로 참이어야 하는 것들이고, 깨지면 코드 버그다.

- 빈 병변 = control (bit-exact)
- 단조성 `D(A∪B) ≥ max(D(A), D(B))` — 1,431쌍 전부 위반 0건
- α 고정 시 `max(r_lesion / r_control) = 1.00000000` (병변이 influence를 늘릴 수 없음)
- 엣지 제거가 열 0으로 만들기와 정확히 일치 (`run_edges.py`가 대조, 불일치 시 중단)
