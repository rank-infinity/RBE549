data_path = "Nehal_p2/Phase1/P2Data"
input_file = data_path + "/matching1.txt"
output_file = data_path + "/matching1_dedup.txt"

seen = set()
kept_lines = []

total_lines = 0
removed = 0

with open(input_file, "r") as f:
    header = next(f)
    kept_lines.append(header)

    for line in f:
        parts = line.split()
        if len(parts) < 5:
            continue

        total_lines += 1
        key = (parts[3], parts[4])

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        kept_lines.append(line)

with open(output_file, "w") as f:
    f.writelines(kept_lines)

print(f"Total entries    : {total_lines}")
print(f"Unique entries   : {len(seen)}")
print(f"Removed entries  : {removed}")
print(f"Clean file saved as: {output_file}")
