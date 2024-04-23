import torch, argparse, sys, os
from torchvision.datasets import DatasetFolder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from carlschader_ml_utils.embedding_utils import _heap_vote

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Given an evaluation dataset folder of embeddings, classify them using knn against another dataset folder of embeddings.')
parser.add_argument('-e', '--embedding_eval_set_path', required=True, type=str, help='Path to the folder of embeddings')
parser.add_argument('-d', '--embedding_dataset_path', required=True, type=str, help='Path to the image to classify')
parser.add_argument('-k', type=int, default=5, help='Number of closest images to return')
parser.add_argument('-tb', '--test-batch-size', type=int, default=8, help='Batch size for the test set')
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for the dataset')
args = parser.parse_args()

dataset_path = args.embedding_dataset_path
test_set_path = args.embedding_eval_set_path
k = args.k
batch_size = args.batch_size
test_batch_size = args.test_batch_size

dataset = DatasetFolder(dataset_path, loader=lambda x: torch.load(x), extensions=('.pt', '.pth'))
test_set = DatasetFolder(test_set_path, loader=lambda x: torch.load(x), extensions=('.pt', '.pth'))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

correct = 0
total = 0
test_batch_count = 0
test_batch_total = len(testloader)

for _, (test_batch, test_classes) in enumerate(testloader):
    test_batch = test_batch.to(device)
    max_heaps = [[] for _ in range(test_batch.shape[0])]
    for emb_batch_count, (embeddings, targets) in enumerate(dataloader):
        embeddings = embeddings.to(device)
        distance_matrix = torch.cdist(test_batch, embeddings)
        
        for i, distance_row in enumerate(distance_matrix):
            for j, distance in enumerate(distance_row):
                target = dataset.classes[targets[j]]
                if len(max_heaps[i]) < k:
                    max_heaps[i].append((distance, target))
                else:
                    max_distance, _ = max_heaps[i][k - 1]
                    if distance < max_distance:
                        max_heaps[i][k - 1] = (distance, target)
                        max_heaps[i].sort(key=lambda x: x[0])
        print(f'Batch {emb_batch_count} / {len(dataloader)}', end='\r')

    batch_correct = 0
    batch_total = 0
    votes = [_heap_vote(heap) for heap in max_heaps]
    for i, vote in enumerate(votes):
        if vote[1] == test_set.classes[test_classes[i]]:
            batch_correct += 1
        batch_total += 1

    correct += batch_correct
    total += batch_total

    test_batch_count += 1

    print("Test Batches: ", test_batch_count, "/", test_batch_total)
    print("Batch accuracy: ", batch_correct / batch_total)
    print("Total accuracy: ", correct / total)

