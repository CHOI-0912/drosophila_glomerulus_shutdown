#!/usr/bin/env bash
# 환경 구축: upstream 라이브러리를 클론하고, 가드 패치를 적용하고, 의존성을 설치한다.
#
#   bash setup.sh
#
# 이 저장소는 분석 코드만 담는다. 시뮬레이터(InfluenceCalculator)는 upstream에서
# 클론하고, BANC 데이터는 용량 때문에 git에 넣지 않는다 (아래 안내).
set -euo pipefail

UPSTREAM_URL="https://github.com/DrugowitschLab/ConnectomeInfluenceCalculator.git"
# 이 분석이 검증된 정확한 커밋. 움직이는 main을 따라가지 않는다.
UPSTREAM_PIN="57cc08cacfb42cbe6fa68cc2a2f25e5924d38b34"
LIB_DIR="ConnectomeInfluenceCalculator"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- 1. 라이브러리
if [ -d "$LIB_DIR/.git" ]; then
  echo "[1/4] $LIB_DIR 이미 존재 — 클론 건너뜀"
else
  echo "[1/4] upstream 라이브러리 클론 …"
  git clone "$UPSTREAM_URL" "$LIB_DIR"
fi

echo "      커밋 $UPSTREAM_PIN 으로 고정"
git -C "$LIB_DIR" fetch --quiet origin
git -C "$LIB_DIR" checkout --quiet "$UPSTREAM_PIN"

# ---------------------------------------------------------------- 2. 가드 패치
# 라이브러리의 조용한 실패 두 가지를 막는다. 자세한 근거는 CHANGES.md 참조:
#   (a) GMRES 미수렴을 검사하지 않고 반환 — 미수렴 해는 NaN도 이상치도 없어서
#       사후에 걸러낼 수 없다.
#   (b) `eig_val_largest > 0` 가드가 무의미 — nilpotent(feedforward) W에서 SLEPc가
#       0이 아니라 작은 양수(~3.7e-5)를 반환해, alpha가 ~1e4배로 폭발한다.
#
# 이 스크린 자체는 이 패치에 의존하지 않는다 (glom_screen/engine.py가
# calculate_influence를 우회하고 자체 solve + 수렴 검사를 갖는다).
# 그래도 라이브러리를 직접 쓸 때를 위해 적용해 둔다.
echo "[2/4] 라이브러리 가드 패치 적용 …"
if git -C "$LIB_DIR" apply --check ../library-guards.patch 2>/dev/null; then
  git -C "$LIB_DIR" apply ../library-guards.patch
  echo "      적용됨"
else
  echo "      이미 적용되었거나 적용 불가 — 건너뜀"
fi

# ---------------------------------------------------------------- 3. 의존성
echo "[3/4] 의존성 확인 …"
MISSING=""
for pkg in petsc4py slepc4py pandas numpy scipy pyarrow bidict; do
  python -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
done

if [ -n "$MISSING" ]; then
  cat <<EOF

      누락:$MISSING

      petsc4py / slepc4py 는 pip으로 깔기 까다롭다. conda 권장:

          conda create -n fly_neuron python=3.13
          conda activate fly_neuron
          conda install -c conda-forge petsc4py slepc4py pandas numpy scipy pyarrow
          pip install bidict
          pip install -e ./$LIB_DIR

      그 뒤 이 스크립트를 다시 실행할 것.

EOF
  exit 1
fi
python -c "import InfluenceCalculator" 2>/dev/null || pip install -e "./$LIB_DIR"
echo "      OK"

# ---------------------------------------------------------------- 4. 데이터
# BANC 커넥톰 데이터는 347MB라 GitHub 100MB 파일 제한을 넘는다. 커밋하지 않는다.
echo "[4/4] BANC 데이터 확인 …"
NEED=""
for f in banc_888_edgelist_simple_v2.feather banc_888_meta.feather; do
  [ -f "$f" ] || NEED="$NEED $f"
done

if [ -n "$NEED" ]; then
  cat <<EOF

      누락:$NEED

      이 두 파일을 저장소 루트($ROOT)에 놓아야 한다.
      용량이 커서(각각 292MB / 55MB) git으로 배포하지 않는다.

        banc_888_edgelist_simple_v2.feather   엣지 리스트 (pre, post, count, …)
        banc_888_meta.feather                 뉴런 메타 (root_id, cell_type, cell_class, …)

      neuron.csv 와 orn.txt 는 이 저장소에 포함되어 있으므로 다시 만들 필요 없다.
      (neuron.csv를 다시 만들려면: python make_silences.py)

EOF
  exit 1
fi
echo "      OK"

cat <<EOF

준비 완료. 실행 순서:

    python -m glom_screen.check        # 불변식 검증 — 먼저 통과해야 함
    python -m glom_screen.run_edges    # ★ 엣지 수준, 최종 결과      (~35초)
    python -m glom_screen.alpha_probe  #   왜 alpha를 고정해야 하는가 (~3분)
    python -m glom_screen.run          #   뉴런 수준 (대조군)          (~40초)
    python -m glom_screen.combos       #   쌍 1,431개                  (~7분)
    python -m glom_screen.report       #   순위 + 강건성 진단

자세한 내용은 glom_screen/README.md
EOF
