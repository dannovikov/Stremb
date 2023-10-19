from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import csv

distances_file = r"G:\3k_ea_training_dists.csv"
map_file = r"G:\3k_ea_training_dists.csv.map"
outfile = r"G:\subtype_dists.csv"

def parse(map_int_to_seqid, line):
    source, target, dist = line.split(',')
    source_type = map_int_to_seqid[int(source)].split('.')[0]
    target_type = map_int_to_seqid[int(target)].split('.')[0]
    return source_type, target_type, float(dist)

def process_chunk(map_int_to_seqid, chunk):
    local_counts = {}
    local_distances = {}
    for line in chunk:
        subtype1, subtype2, dist = parse(map_int_to_seqid, line)
        local_counts[subtype1] = local_counts.get(subtype1, 0) + 1
        local_counts[subtype2] = local_counts.get(subtype2, 0) + 1
        if subtype1 not in local_distances:
            local_distances[subtype1] = {}
        if subtype2 not in local_distances[subtype1]:
            local_distances[subtype1][subtype2] = 0
        local_distances[subtype1][subtype2] += dist
    return local_counts, local_distances

if __name__ == '__main__':
    map_int_to_seqid = {}
    with open(map_file, 'r') as f:
        for line in f:
            i, seqid = line.strip().split(',')
            map_int_to_seqid[int(i)] = seqid

    chunk_size = 10000
    results = []
    with Pool(processes=cpu_count()) as pool:
        chunk = []
        with open(distances_file, 'r') as f:
            for i, line in tqdm(enumerate(f), total=32394414917):
                if i == 0:
                    continue
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    result = pool.apply_async(process_chunk, [map_int_to_seqid, chunk])
                    results.append(result)
                    chunk = []
            if chunk:
                result = pool.apply_async(process_chunk, [map_int_to_seqid, chunk])
                results.append(result)
        pool.close()
        pool.join()

    subtype_counts = {}
    subtype_distances = {}
    for result in results:
        local_counts, local_distances = result.get()
        for subtype, count in local_counts.items():
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + count
        for subtype1, targets in local_distances.items():
            if subtype1 not in subtype_distances:
                subtype_distances[subtype1] = {}
            for subtype2, dist in targets.items():
                if subtype2 not in subtype_distances[subtype1]:
                    subtype_distances[subtype1][subtype2] = 0
                subtype_distances[subtype1][subtype2] += dist

    for subtype1 in subtype_counts.keys():
        for subtype2 in subtype_counts.keys():
            if subtype2 in subtype_distances.get(subtype1, {}):
                subtype_distances[subtype1][subtype2] /= (subtype_counts[subtype1] + subtype_counts[subtype2])

    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        for subtype1 in subtype_counts.keys():
            for subtype2 in subtype_counts.keys():
                writer.writerow([subtype1, subtype2, subtype_distances.get(subtype1, {}).get(subtype2, 0.0)])
