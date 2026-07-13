from pathlib import Path
import pandas as pd


META_PATH = Path("banc_888_meta.feather")
EDGE_PATH = Path("banc_888_edgelist_simple_v2.feather")
OUTPUT_PATH = Path("neuron.csv")

# Codex와 비슷하게 약한 연결을 제외하려면 3
# 모든 연결을 포함하려면 1로 변경
MIN_SYNAPSE_COUNT = 3


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"{description} 컬럼을 찾지 못했습니다.\n"
        f"현재 컬럼: {df.columns.tolist()}"
    )


# 파일 읽기
meta = pd.read_feather(META_PATH)
edges = pd.read_feather(EDGE_PATH)

# 실제 컬럼명 자동 탐색
id_col = find_column(
    meta,
    ["banc_888_id", "root_id", "id", "segment_id"],
    "뉴런 ID",
)

cell_type_col = find_column(
    meta,
    ["cell_type", "celltype"],
    "cell_type",
)

pre_col = find_column(
    edges,
    ["pre", "pre_root_id", "pre_id"],
    "presynaptic 뉴런 ID",
)

post_col = find_column(
    edges,
    ["post", "post_root_id", "post_id"],
    "postsynaptic 뉴런 ID",
)

count_col = find_column(
    edges,
    ["count", "weight", "syn_count", "synapse_count"],
    "시냅스 개수",
)

# ORN만 선택
orn = meta[[id_col, cell_type_col]].copy()
orn[cell_type_col] = orn[cell_type_col].fillna("").astype(str)

orn = orn[
    orn[cell_type_col].str.startswith("ORN_")
].copy()

# ORN_DM1 -> DM1
orn["glomerulus"] = orn[cell_type_col].str.removeprefix("ORN_")

# ID 자료형 통일
orn[id_col] = orn[id_col].astype(str)
edges[pre_col] = edges[pre_col].astype(str)
edges[post_col] = edges[post_col].astype(str)

# ORN이 presynaptic인 연결만 선택
connections = edges[
    edges[pre_col].isin(orn[id_col])
].copy()

# 약한 연결 제외
connections = connections[
    connections[count_col] >= MIN_SYNAPSE_COUNT
].copy()

# 각 ORN 연결에 glomerulus 이름 추가
connections = connections.merge(
    orn[[id_col, "glomerulus"]],
    left_on=pre_col,
    right_on=id_col,
    how="inner",
)

# 같은 downstream 뉴런 중복 제거
glomerulus_neurons = (
    connections[["glomerulus", post_col]]
    .drop_duplicates()
    .sort_values(["glomerulus", post_col])
)

# glomerulus당 한 행, 뉴런 ID들은 쉼표로 연결
result = (
    glomerulus_neurons
    .groupby("glomerulus")[post_col]
    .agg(lambda ids: ",".join(ids))
    .reset_index()
    .rename(columns={post_col: "neurons"})
)

# 상대경로 neuron.csv로 저장
result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"저장 완료: {OUTPUT_PATH.resolve()}")
print(f"glomerulus 수: {len(result)}")
print(result.head(10).to_string(index=False))