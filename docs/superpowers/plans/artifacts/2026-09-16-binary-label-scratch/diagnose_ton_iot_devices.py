from data.iot_datasets import _load_ton_iot_device_dataframes

device_dataframes = _load_ton_iot_device_dataframes("./ZT-Fedtrust Datasets")
for name in sorted(device_dataframes):
    df = device_dataframes[name]
    counts = df["type"].value_counts()
    normal = int(counts.get("normal", 0))
    attack = int(len(df) - normal)
    print(f"{name}: total_rows={len(df)} normal={normal} attack={attack} attack_pct={100*attack/len(df):.1f}%", flush=True)
