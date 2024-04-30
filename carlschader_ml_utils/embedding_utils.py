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
    import torch, argparse, sys, os
    from torchvision.datasets import DatasetFolder
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Given an evaluation dataset folder of embeddings, classify them using euclidean distance against another dataset folder of embeddings.')
    parser.add_argument('-e' ,'--embedding_eval_set_path', required=True, type=str, help='Path to the folder of embeddings')
    parser.add_argument('-d', '--embedding_dataset_path', required=True, type=str, help='Path to the image to classify')
    parser.add_argument('-t', '--test-batch-size', type=int, default=32, help='Batch size for the test set')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for the dataset')
    parser.add_argument('-m', '--metric', type=str, default='euclidean', help='Distance metric to use')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='Output file to write results to')
    args = parser.parse_args()

    dataset_path = args.embedding_dataset_path
    test_set_path = args.embedding_eval_set_path
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    metric = args.metric

    def cos_sim(a, b):
        anorm = a / a.norm(dim=1).unsqueeze(1)
        bnorm = b / b.norm(dim=1).unsqueeze(1)
        return 1 - torch.mm(anorm, bnorm.T)

    distance_function = None
    if metric == 'euclidean':
        distance_function = torch.cdist
    elif metric == 'cosine':
        distance_function = cos_sim
    else:
        raise ValueError('Invalid distance metric, valid options are euclidean and cosine')

    dataset = DatasetFolder(dataset_path, loader=lambda x: torch.load(x), extensions=('.pt', '.pth'))
    test_set = DatasetFolder(test_set_path, loader=lambda x: torch.load(x), extensions=('.pt', '.pth'))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    correct = 0
    total = 0
    test_batch_count = 0
    test_batch_total = len(testloader)

    for _, (test_batch, test_targets) in enumerate(testloader):
        test_batch = test_batch.to(device)
        test_targets = test_targets.to(device)
        min_distances = None
        min_targets = None
        for emb_batch_count, (embeddings, targets) in enumerate(dataloader):
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            # distance_matrix = torch.cdist(test_batch, embeddings)
            distance_matrix = distance_function(test_batch, embeddings)
            batch_mins, min_indices = torch.min(distance_matrix, dim=1)
            batch_min_targets = targets[min_indices]

            if min_distances is None:
                min_distances = batch_mins
                min_targets = batch_min_targets
            else:
                min_indices = torch.lt(batch_mins, min_distances)
                min_distances = torch.where(min_indices, batch_mins, min_distances)
                min_targets = torch.where(min_indices, batch_min_targets, min_targets)
            print(f'Batch {emb_batch_count} / {len(dataloader)}', end='\r')

        min_classes = [dataset.classes[min_targets[i]] for i in range(len(test_targets))]
        test_classes = [test_set.classes[test_targets[i]] for i in range(len(test_targets))]
        correct_vec = torch.tensor([1 if dataset.classes[min_targets[i]] == test_set.classes[test_targets[i]] else 0 for i in range(len(test_targets))])
        batch_correct = torch.sum(correct_vec).item()
        batch_total = len(test_targets)
        
        correct += batch_correct
        total += batch_total

        test_batch_count += 1

        print("Test Batches: ", test_batch_count, "/", test_batch_total)
        print("Batch accuracy: ", batch_correct / batch_total)
        print("Total accuracy: ", correct / total)


