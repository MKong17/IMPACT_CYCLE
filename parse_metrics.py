import re
from collections import defaultdict

# Read the entire log file
with open('/cvhci/temp/wkong/IMPACT_VQA/paper5_vidor_full_f3.log', 'r') as f:
    lines = f.readlines()

# Pattern to match result lines
pattern = r'variant=([\w_]+).*claim_acc=([\d.]+)->([\d.]+)\s+vote_dir_acc=([\d.]+)\s+false_claim_reduction=(\d+)\s+hit=(\d+)'

# Dictionary to store metrics by variant
variants = defaultdict(lambda: {
    'claim_acc': [],
    'vote_dir_acc': [],
    'false_claim_reduction': [],
    'hit': []
})

# Parse all result lines
for line in lines:
    if 'frame' in line and 'done' in line and 'variant=' in line:
        match = re.search(pattern, line)
        if match:
            variant, before_acc, after_acc, vote_acc, fcr, hit = match.groups()
            variants[variant]['claim_acc'].append(float(after_acc))
            variants[variant]['vote_dir_acc'].append(float(vote_acc))
            variants[variant]['false_claim_reduction'].append(int(fcr))
            variants[variant]['hit'].append(int(hit))

# Calculate and display results
variant_order = ['backbone_only', 'single_turn', 'single_turn_caption', 'single_turn_multi', 'full_cycle']

print("| Variant | Claim Accuracy | Vote Dir Accuracy | False Claim Reduction | Hit |")
print("|---------|----------------|-------------------|----------------------|-----|")

for variant in variant_order:
    if variant in variants:
        data = variants[variant]
        avg_claim = sum(data['claim_acc']) / len(data['claim_acc']) if data['claim_acc'] else 0
        avg_vote = sum(data['vote_dir_acc']) / len(data['vote_dir_acc']) if data['vote_dir_acc'] else 0
        avg_fcr = sum(data['false_claim_reduction']) / len(data['false_claim_reduction']) if data['false_claim_reduction'] else 0
        avg_hit = sum(data['hit']) / len(data['hit']) if data['hit'] else 0
        
        print(f"| {variant} | {avg_claim:.4f} | {avg_vote:.4f} | {avg_fcr:.4f} | {avg_hit:.4f} |")