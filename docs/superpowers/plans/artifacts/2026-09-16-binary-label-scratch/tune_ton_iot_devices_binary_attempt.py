import sys
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

attempt_label = sys.argv[1] if len(sys.argv) > 1 else "attempt"
cfg = load_config('configs/iot/ton_iot_devices_binary.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'ton_iot_devices (binary) baseline {attempt_label}: accuracy={acc:.2f}%', flush=True)
