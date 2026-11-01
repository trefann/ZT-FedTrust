from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

runs = [
    ('configs/iot/nbaiot_binary.yaml', 'nbaiot (binary)'),
    ('configs/iot/ton_iot_network_binary.yaml', 'ton_iot_network (binary)'),
]

for path, name in runs:
    cfg = load_config(path)
    for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
        result = ExperimentRunner(cfg).run(mode)
        acc = result.history[-1]['test_accuracy'] * 100
        print(f'{name} {mode}: accuracy={acc:.2f}%', flush=True)
