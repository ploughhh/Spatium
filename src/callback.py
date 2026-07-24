import torch
import pytorch_lightning as pl

class SaveEmbeddingCallback(pl.Callback):
    def __init__(self, save_path, dataloader, target_epoch=10, task=None):

        super().__init__()
        self.save_path = save_path
        self.dataloader = dataloader
        self.target_epoch = target_epoch
        self.task = task

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        if current_epoch != self.target_epoch:
            return

        pl_module.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in self.dataloader:
                batch_device = {k: v.to(pl_module.device) for k, v in batch.items() if torch.is_tensor(v)}
                if 'x' in batch_device:
                    x = batch_device['x']
                    attention_mask = (x == 0) if self.task is None else (x == 0)
                else:
                    x = None
                    attention_mask = None

                if x is not None:
                    emb = pl_module.get_embedding(x, attention_mask)
                    all_embeddings.append(emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        torch.save(embeddings, self.save_path)
        print(f"[Callback] Saved embedding at epoch {current_epoch} to {self.save_path}")
        pl_module.train()