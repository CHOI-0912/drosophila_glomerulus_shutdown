#!/usr/bin/env bash
# 환경 구축: upstream 라이브러리를 클론하고, 가드 패치를 적용하고, 의존성을 설치한다.
#
#   bash setup.sh
#
# 이 저장소는 분석 코드만 담는다. 시뮬레이터(InfluenceCalculator)는 upstream에서
# 클론하고, BANC 데이터(347MB)는 Lee Lab 공개 버킷에서 내려받는다.
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
# BANC 커넥톰 데이터는 347MB라 GitHub 100MB 파일 제한을 넘는다. 커밋하지 않고,
# Lee Lab의 공개 GCS 버킷에서 받는다 (upstream README가 안내하는 경로).
# 받은 파일은 크기와 sha256으로 검증한다. 잘린 다운로드가 조용히 통과하면 안 된다.
BUCKET="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888"

EDGELIST_FILE="banc_888_edgelist_simple_v2.feather"
EDGELIST_SIZE=305250378
EDGELIST_SHA="363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb"

META_FILE="banc_888_meta.feather"
META_SIZE=57550610
META_SHA="819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5"

file_size() { wc -c < "$1" | tr -d ' '; }

fetch_data() {
  local file="$1" size="$2" sha="$3"

  if [ -f "$file" ] && [ "$(file_size "$file")" = "$size" ]; then
    echo "      $file 이미 있음 — 건너뜀"
    return 0
  fi

  if [ -f "$file" ]; then
    echo "      $file 크기가 안 맞음 — 다시 받는다"
    rm -f "$file"
  fi

  echo "      $file 내려받는 중 … ($((size / 1024 / 1024))MB)"
  # .part로 받고 검증 후에만 제자리로 옮긴다. 중간에 끊겨도 반쪽 파일이 남지 않는다.
  curl -fL --retry 3 --retry-delay 2 -o "$file.part" "$BUCKET/$file"

  local got
  got="$(file_size "$file.part")"
  if [ "$got" != "$size" ]; then
    rm -f "$file.part"
    echo "      실패: $file 크기가 $size 여야 하는데 $got 였다." >&2
    return 1
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    local got_sha
    got_sha="$(sha256sum "$file.part" | cut -d' ' -f1)"
    if [ "$got_sha" != "$sha" ]; then
      rm -f "$file.part"
      echo "      실패: $file 의 sha256이 다르다." >&2
      echo "            기대: $sha" >&2
      echo "            실제: $got_sha" >&2
      return 1
    fi
  fi

  mv "$file.part" "$file"
  echo "      OK ($file)"
}

echo "[4/4] BANC 데이터 …"
if ! command -v curl >/dev/null 2>&1; then
  echo "      curl이 없다. 설치하거나, 아래 두 파일을 직접 받아 $ROOT 에 놓을 것:" >&2
  echo "        $BUCKET/$EDGELIST_FILE" >&2
  echo "        $BUCKET/$META_FILE" >&2
  exit 1
fi

fetch_data "$EDGELIST_FILE" "$EDGELIST_SIZE" "$EDGELIST_SHA"
fetch_data "$META_FILE"     "$META_SIZE"     "$META_SHA"

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
