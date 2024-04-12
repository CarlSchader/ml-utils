import torch, argparse, sys, os
from torchvision.datasets import DatasetFolder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from carlschader_ml_utils.embedding_utils import _heap_vote

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Given a target dataset folder of embeddings, classify them using euclidean distance against another dataset folder of embeddings.')
parser.add_argument('embedding_test_set_path', type=str, help='Path to the folder of embeddings')
parser.add_argument('embedding_dataset_path', type=str, help='Path to the image to classify')
parser.add_argument('-tb', '--test-batch-size', type=int, default=8, help='Batch size for the test set')
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for the dataset')
args = parser.parse_args()

dataset_path = args.embedding_dataset_path
test_set_path = args.embedding_test_set_path
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

for _, (test_batch, test_targets) in enumerate(testloader):
    test_batch = test_batch.to(device)
    min_distances = None
    min_targets = None
    for emb_batch_count, (embeddings, targets) in enumerate(dataloader):
        embeddings = embeddings.to(device)
        distance_matrix = torch.cdist(test_batch, embeddings)
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

    min_classes = dataset.classes[min_targets]

    batch_correct = torch.sum(torch.eq(min_classes, test_set.classes[test_targets]))
    batch_total = len(test_targets)
    
    correct += batch_correct
    total += batch_total

    test_batch_count += 1

    print("Test Batches: ", test_batch_count, "/", test_batch_total)
    print("Batch accuracy: ", batch_correct / batch_total)
    print("Total accuracy: ", correct / total)

