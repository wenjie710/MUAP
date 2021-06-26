import torch


def mean_var(result):
    mean = torch.mean(result, 0)
    var = torch.var(result, 0)
    return mean, var


result23 = torch.tensor([41.5, 53.2, 66.5, 72.3, 40.7, 51.3, 64.8, 70.0, 41.7, 52.9, 67.5, 72.6, 40.7, 50.7, 64.2, 69.9]).reshape(-1, 4)
all_results = [result23]

for result in all_results:
    mean, var = mean_var(result)
    print(mean, var)
