# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm


class Predictor:

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def run_inference(self, data_loader):
        self.model.eval()
        inference_result = []
        for batch in tqdm(data_loader, total=len(data_loader)):
            with torch.no_grad():
                inference_result += self.inference_one_batch(batch)
        return inference_result

    def inference_one_batch(self, batch):
        outputs = self.model(batch['image'].to(
            self.device, dtype=torch.float32))
        predictions = []
        for sample_id, gt_text, output in zip(batch['id'], batch['text'], outputs):
            predictions.append({
                'id': sample_id,
                'raw_output': output.detach().cpu(),
                'gt_text': gt_text,
            })
        return predictions
