# 변경 내역 (2026-07-13)

`InfluenceCalculator`의 silence(lesion) 계산에 정규화 누락 버그가 있다는 제보를 검증하고,
그 과정에서 발견한 실제 결함들을 수정했다.

---

## 1. 제보된 버그는 존재하지 않았다 (수정 없음)

**제보 내용:** silence 경로에서 `_normalize_W()`가 호출되지 않아, `(αW − I)r = −s` 대신
정규화도 `−I`도 없는 `W⁽⁻ʲ⁾r = −s`를 풀고 있다.

**검증 결과: 사실이 아니다.** `calculate_influence`에서 `_normalize_W`는 `if/else` **바깥**에 있어
두 분기 모두에서 실행된다.

```python
if len(silenced_neurons) > 0:
    W_norm = self._set_columns_to_zero(silenced_indices)
else:
    W_norm = self.W.copy()

self._normalize_W(W_norm)          # ← if/else 바깥. 양쪽 모두 통과한다.
influence_vec = self._solve_lin_system(W_norm, -seed_vec)
```

그리고 `_normalize_W`는 `scale(α)` 후 `shift(-1.0)`을 수행하므로 `−I`도 정확히 들어간다.
즉 silence 시 실제로 푸는 식은 **`(α_lesion · W⁽⁻ʲ⁾ − I) r = −s`** 이며, 이는 제보가 제시한
두 선택지 중 **2번(lesion된 행렬의 고유값으로 α 재계산)** 에 해당한다. "둘 중 어느 것도 아니다"는
틀렸다.

파생 주장도 함께 무너진다:
- `W⁽⁻ʲ⁾`가 특이행렬일 것이다 → **아니다.** `αW`의 최대 실수부 고유값이 `lambda_max`(0.99)로
  고정되므로 `αW − I`의 모든 고유값은 실수부 ≤ −0.01 이다. 비특이이며 안정적이다.

확인 범위: 현재 소스, `build/lib` 사본, git `HEAD`(= `origin/main`), 그리고 파일이 처음 추가된
최초 커밋 `fa3a33f`. **전부 동일하게 `_normalize_W`를 무조건 호출한다. 애초에 깨진 적이 없다.**

### 단, α 재정규화는 실재하는 문제이고 — 처음 평가보다 훨씬 심각하다

lesion 후 α를 다시 계산하므로 남은 연결이 up-scale되어 제거 효과를 보상한다.
**나는 처음에 이걸 "무시 가능"이라고 판단했는데, 그건 틀렸다.**

처음 근거는 α 드리프트의 절대 크기였다:

| | λ_max(W) | α |
|---|---|---|
| control | 2191.12 | 4.5182e-04 |
| sensory 13,710개(9.1%) silence | 2176.36 | 4.5489e-04 |

→ 남은 시냅스가 **+0.68%** 증폭. 확실히 작다.

**그런데 잘못된 것을 쟀다.** 중요한 건 α 드리프트의 절대 크기가 아니라 **그것이 측정하려는 신호를
얼마나 상쇄하는가**다. 이후 glomerulus 병변 스크린(`glom_screen/alpha_probe.py`, 54개 조건 실측):

| | D_mag_sum 범위 |
|---|---|
| α 고정 | 0.016 ~ **0.589** |
| α 재계산 (라이브러리 기본값) | 0.000 ~ 0.023 |

- α 재계산이 **신호의 95.6%를 흡수**한다.
- 남은 잔여물은 진짜 효과와 **Spearman = −0.06** — 상관이 없다.
- 6개 조건은 D가 **음수**(병변이 influence를 증가시킴)로 나와 0으로 clip된다.

이유가 고약하다: 병변이 중요할수록 λ_max가 많이 떨어지고, 그래서 α 보정이 커져서 효과를 정확히
상쇄한다. 즉 **보상량이 신호에 비례한다.** DL2d는 진짜 D=0.589인데 재계산 시 0.0034가 된다.

**결론: lesion 연구에서 라이브러리 기본 동작(α 재계산)을 쓰면 안 된다.** 병변 크기가 작을수록
(신호가 작을수록) 이 문제는 더 심해진다.

라이브러리는 수정하지 않기로 했으므로(사용자 결정), `glom_screen/engine.py`가
`calculate_influence`를 우회해 **온전한 W에서 α를 한 번만 구하고 모든 조건에 고정**한다.
부수 효과로 `r_lesion ≤ r_control`이 모든 뉴런에서 정확히 성립한다(경로 제거만 일어나므로).

---

## 2. 실제로 고친 것

### (a) `_solve_lin_system`: GMRES 수렴 검사 추가

**문제:** `ksp.solve()` 후 `getConvergedReason()`을 확인하지 않고 그대로 반환했다.
미수렴 해는 **사후에 걸러낼 수 없다.** NaN도 inf도 이상치도 생기지 않고, 오차가 전체 뉴런에
고르게 퍼진 "그럴듯한" 벡터가 나오기 때문이다. 실측:

| Krylov dim | rel_resid | NaN/inf | Spearman | top-1000 일치 | e배 이상 오차 |
|---|---|---|---|---|---|
| 5 | 8.3e-02 | 0 / 0 | 0.904 | 629/1000 | 115,668 |
| 10 | 6.8e-03 | 0 / 0 | **0.991** | **999/1000** | **34,664** |
| (정상) | 9.8e-06 | 0 / 0 | 1.000 | 1000/1000 | 0 |

`krylov=10` 행이 위험하다. 어떤 sanity check도 통과하지만 뉴런의 23%가 e배 이상 틀렸다.
잔차라는 스칼라 하나만 보면 100% 판별되고, PETSc가 이미 계산해둔 값이다.

```python
reason = ksp.getConvergedReason()
if reason < 0:
    raise RuntimeError(...)
```

반환 벡터 `x`는 손대지 않는다. 숫자는 바뀌지 않고, "쓰레기를 받을지 예외를 받을지"만 바뀐다.

### (b) `_normalize_W`: `eig_val_largest > 0` 가드를 상대 하한으로 교체 — **실제 버그 수정**

기존 코드는 `if eig_val_largest > 0:` 일 때만 스케일링했다. 이 가드는 **작동하지 않는다.**

feedforward(비순환) 커넥톰의 W는 **nilpotent**이라 참 고유값이 정확히 0이다. 그런데 nilpotent
행렬은 **defective**이고, defective 행렬의 고유값 오차는 Jordan 블록 크기 m에 대해 `eps^(1/m)`
수준이다. 그래서 SLEPc는 0이 아니라 **작은 양수**를 반환한다. 실측: 4노드 체인에서 `3.67e-05`.

결과적으로 `> 0` 가드를 통과해버리고:

```
alpha = 0.5 / 3.67e-05 = 13,629
→ influence 점수: 1.36e+04, 1.86e+08, 2.53e+12     (정답은 전부 1.0)
```

**경고도 에러도 NaN도 없이 1e12짜리 쓰레기가 나온다.** 정확히 (a)에서 우려한 조용한 실패다.

수정: 고유값이 `||W||_F`에 비례하는 하한(`_EIG_REL_FLOOR = 1e-3`)을 넘어야 스케일링하고,
못 넘으면 `RuntimeWarning`을 띄운 뒤 스케일링을 건너뛴다(이 경우 `W − I`는 여전히 비특이·안정
이며 해는 정확한 Neumann 합이다 — 에러가 아니라 경고가 맞다).

하한값 근거 — 실측 `eig / ||W||_F`:

| | 비율 |
|---|---|
| nilpotent 노이즈 | 2.1e-05 |
| **하한 1e-3** | — |
| BANC | 0.087 |
| C. elegans | 0.175 |

노이즈보다 ~50배 위, BANC보다 ~90배 아래. 양방향 마진이 충분하다.

> 참고: 이건 가상의 시나리오가 아니다. 새 테스트에서 3-사이클의 허브 뉴런을 silence하자
> 남은 그래프가 비순환이 되어 이 경고가 실제로 발동했다. **lesion이 순환 구조를 파괴할 수 있고,
> 그 경우 기존 코드는 조용히 폭발했다.**

### (c) silencing 테스트 5개 추가 (기존 0개)

`tests/`에 silencing을 다루는 테스트가 **하나도 없었다.** 추가한 것:

- `test_silencing_output_free_neuron_matches_no_silencing` — **1번 제보에 대한 회귀 테스트.**
  출력이 없는 뉴런을 silence하면 W가 그대로이므로 결과가 no-silence와 정확히 일치해야 한다.
  두 분기가 `_normalize_W`를 공유하지 않으면 자릿수 단위로 어긋난다.
- `test_silencing_blocks_downstream_propagation` — silence된 뉴런을 통해서만 도달 가능한
  하류 뉴런의 influence가 0이 되는지.
- `test_silencing_only_seeds_is_a_no_op` — seed는 silence 대상에서 제외된다(README의 주장).
- `test_silencing_changes_influence_on_real_connectome` — 실제 커넥톰에서 허브를 silence하면
  값이 유한하게 유지되면서 실제로 변하는지.
- `test_feedforward_W_warns_and_does_not_blow_up` — (b)의 회귀 테스트. 경고가 뜨고 점수가
  1e12가 아니라 O(1)로 유지되는지.

---

## 3. 기존 결과에 미치는 영향: **없음**

두 가드 모두 실제 데이터에서는 no-op이다.

- (a) 반환 벡터를 변경하지 않는다(설계상 불가능). BANC는 20 Krylov step에서 잔차 1e-5 도달,
  PETSc 기본 한도는 10,000 — 마진 500배.
- (b) BANC의 `eig/||W||_F = 0.087`로 하한(1e-3)을 87배 상회. 분기 결과가 기존과 동일하다.

검증: BANC 전체 파이프라인(150,299 뉴런, seed 1,934개, silenced 18,412개)을
`-W error::RuntimeWarning`으로 실행 — 경고·예외 없이 1초 만에 완료, 전 값 유한.

```
built W: 150299 neurons  (8s)
seeds=1934  silenced=18412
solve OK, no warning, no exception  (1s)
  finite=True  nonzero=128702/150299
  min=0.000e+00  median=6.515e-07  max=6.257e+01
```

테스트: **33 passed** (기존 28 + 신규 5).

이번 변경은 지금 나온 숫자를 고치는 게 아니라, `lambda_max`를 바꾸거나 다른 커넥톰을 넣거나
`signed=True`를 켰을 때 — **마진이 어떻게 될지 모르는 순간** — 조용히 틀린 답을 받지 않기
위한 것이다.

---

## 변경 파일

- `InfluenceCalculator/InfluenceCalculator.py`
  - `import warnings` 추가
  - `InfluenceCalculator._EIG_REL_FLOOR = 1e-3` 클래스 상수 신설
  - `_normalize_W()` — 상대 하한 가드 + `RuntimeWarning`
  - `_solve_lin_system()` — `getConvergedReason()` 검사 + `RuntimeError`
- `tests/test_influence_calculator.py` — silencing 섹션 5개 테스트 추가
