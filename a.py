from InfluenceCalculator import InfluenceCalculator

ic = InfluenceCalculator.from_feather(
    "banc_888_edgelist_simple_v2.feather",
    "banc_888_meta.feather",
    signed=False,
    count_thresh=5,
)
# Define seed category (depending on how neurons are labelled in metadata)
meta_column = 'seed_01'
seed_category = 'olfactory'

# Get seed neuron ids
seed_ids = ic.meta[ic.meta[meta_column] == seed_category].root_id 

# Get neuron ids to inhibit (sensory neurons in this case)
silenced_neurons = ic.meta[
    ic.meta['super_class'].isin(['sensory',
                                 'ascending_sensory'])].root_id

# Calculate influence scores and store them in a Pandas dataframe
influence_df = ic.calculate_influence(seed_ids, silenced_neurons)

print(influence_df.keys())

print("객체 종류:", type(ic))

print("\n=== 메타데이터 ===")
print("shape:", ic.meta.shape)
print("뉴런 ID 수:", ic.meta["root_id"].nunique())
print("열 수:", len(ic.meta.columns))

print("\n=== 주요 뉴런 분류 ===")
print(ic.meta["super_class"].value_counts(dropna=False).head(20))

print("\n=== 내부 속성 ===")
for name, value in vars(ic).items():
    if hasattr(value, "getSize"):
        detail = f"PETSc 행렬, size={value.getSize()}"
    elif hasattr(value, "shape"):
        detail = f"shape={value.shape}"
    else:
        detail = repr(value)
        if len(detail) > 120:
            detail = detail[:120] + "..."

    print(f"{name}: {type(value).__name__}, {detail}")