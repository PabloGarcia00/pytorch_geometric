import os
import json
from pathlib import Path

def aggregate_best_loss(results_dir):
    p = Path(results_dir)
    report = []
    
    for subdir in sorted(p.iterdir()):
        if not subdir.is_dir() or '-' not in subdir.name:
            continue
            
        parts = subdir.name.split('-')
        weather = "No Weather"
        mp = "Unknown"
        
        for part in parts:
            if part.startswith('w='):
                val = part[2:]
                weather = "Weather" if len(val) > 2 else "No Weather"
            if part.startswith('mp='):
                mp = f"MP={part[3:]}"
        
        config_name = f"{weather}, {mp}"
        
        for seed_dir in sorted(subdir.glob('[0-9]*')):
            seed = seed_dir.name
            stats_file = seed_dir / 'val' / 'stats.json'
            if stats_file.exists():
                best_loss = float('inf')
                best_epoch = -1
                
                with open(stats_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'loss' in data:
                                if data['loss'] < best_loss:
                                    best_loss = data['loss']
                                    best_epoch = data['epoch']
                        except:
                            continue
                
                if best_epoch != -1:
                    report.append({
                        "Config": config_name,
                        "Seed": seed,
                        "Best Loss": f"{best_loss:.4f}",
                        "Epoch": best_epoch
                    })
                    
    return report

if __name__ == "__main__":
    results = aggregate_best_loss('results/earne_weather_all_15min_grid_exp1')
    print("| Config | Seed | Best Loss | Epoch |")
    print("| :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['Config']} | {r['Seed']} | {r['Best Loss']} | {r['Epoch']} |")
