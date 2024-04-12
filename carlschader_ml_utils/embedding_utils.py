import torch

def _heap_vote(heap):
    mode_target = heap[0]
    mode_count = 1
    counts = {mode_target: 1}

    for _, target in heap[1:]:
        if target not in counts:
            counts[target] = 0
        counts[target] += 1
        if counts[target] > mode_count:
            mode_target = target
            mode_count = counts[target]

    return mode_target

def knn_classify(input_batch, comparison_batch, comparison_classes, k=20, device=torch.device('cpu')):
    input_batch = input_batch.to(device)
    comparison_batch = comparison_batch.to(device)
    distance_matrix = torch.cdist(input_batch, comparison_batch)

    max_heaps = [[] for _ in range(input_batch.shape[0])]
    for i, distance_row in enumerate(distance_matrix):
        for j, distance in enumerate(distance_row):
            target = comparison_classes[j]
            if len(max_heaps[i]) < k:
                max_heaps[i].append((distance, target))
            else:
                max_distance, _ = max_heaps[i][k - 1]
                if distance < max_distance:
                    max_heaps[i][k - 1] = (distance, target)
                    max_heaps[i].sort(key=lambda x: x[0])

    return [_heap_vote(heap) for heap in max_heaps]

def euclidean_distance_classify(input_batch, comparison_batch, comparison_classes, device=torch.device('cpu')):
    input_batch = input_batch.to(device)
    comparison_batch = comparison_batch.to(device)
    distance_matrix = torch.cdist(input_batch, comparison_batch)
    return comparison_classes[torch.argmin(distance_matrix, dim=1)]

