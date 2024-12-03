import lightning as L
import torch
from torcheval.metrics.functional import r2_score
from torch.optim import Adam


class TrainWrapper(L.LightningModule):
    def __init__(self, cfg, encoder, predictor, decoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        if self.predictor is not None:
            params += list(self.predictor.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        optimizer = Adam(
            params,
            lr=self.cfg.lr
        )
        return optimizer

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and hasattr(self.encoder, 'on_train_start'):
            self.encoder.on_train_start()
    
    def training_step(self, batch, batch_idx):
        total_loss = 0
        for trial in batch:
            latents = self.encoder(trial)
            output = self.decoder(torch.Tensor(latents).to(self.device))
            ground_truth = torch.Tensor(trial.behavior.position).to(self.device)
            loss = torch.nn.functional.mse_loss(output, ground_truth)
            total_loss += loss
        avg_loss = total_loss / len(batch)
        self.log('train_loss', avg_loss, prog_bar=True)
        return avg_loss

    def on_validation_epoch_start(self):
        self.outputs = []
        self.ground_truths = []
    
    def validation_step(self, batch, batch_idx):
        for trial in batch:
            latents = self.encoder(trial)
            output = self.decoder(torch.Tensor(latents).to(self.device))
            ground_truth = torch.Tensor(trial.behavior.position).to(self.device)
            self.outputs.append(output)
            self.ground_truths.append(ground_truth)

    def on_validation_epoch_end(self, phase='val'):
        outputs = torch.cat(self.outputs)
        ground_truths = torch.cat(self.ground_truths)
        r2 = r2_score(outputs, ground_truths)
        self.log(f'{phase}_r2', r2, prog_bar=True)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.on_validation_epoch_end(phase='test')