import subprocess
import csv

samp_values = list(range(50, 100, 5))  # 50 to 100, step 5
str_values = ['full', 'diag']

results = []

for str_val in str_values:
    for samp in samp_values:
        cmd = [
            'python', 'banana_experiment.py',
            '-s', '0',
            '-str', str_val,
            '-sub', 'all',
            '-samp', str(samp),
            '-optim', 'sgd',
            '-opt_prior', 'True'
        ]
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout

        # Extract time1, time2, and time_LA
        try:
            lines = output.strip().split('\n')
            time1 = float([line for line in lines if 'manofold' in line][0].split()[-2])
            time2 = float([line for line in lines if 'parallelization' in line][0].split()[-2])
            time_LA = float([line for line in lines if 'LA' in line][0].split()[-2])
        except Exception as e:
            print(f"Failed to parse output for samp={samp}, str={str_val}")
            print("Output was:\n", output)
            time1, time2, time_LA = None, None, None

        results.append({
            'samp': samp,
            'structure': str_val,
            'time_manifold': time1,
            'time_weights': time2,
            'time_weights_LA': time_LA
        })

# Write to CSV
csv_file = 'timing_results_with_optimization.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['samp', 'structure', 'time_manifold', 'time_weights', 'time_weights_LA'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Results written to {csv_file}")
