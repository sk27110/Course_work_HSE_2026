import torch

class Evaluator:
    def __init__(self, metrics=["top1", "top5"], device="cuda"):
        self.metrics = metrics
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, dataloader):
        model.eval()
        total_samples = 0
        top1_correct = 0
        top5_correct = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)

            if "top1" in self.metrics:
                _, preds = outputs.topk(1, dim=1, largest=True, sorted=True)
                top1_correct += (preds.squeeze(1) == labels).sum().item()

            if "top5" in self.metrics:
                _, preds5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                top5_correct += (preds5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += labels.size(0)

        results = {}
        if "top1" in self.metrics:
            results["top1"] = top1_correct / total_samples
        if "top5" in self.metrics:
            results["top5"] = top5_correct / total_samples

        return results
