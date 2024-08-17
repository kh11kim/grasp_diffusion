import lightning as L
from se3dif.models.loader import *
from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules
import se3dif.losses as losses

class SE3DifModule(L.LightningModule):
    def __init__(self, param:dict):
        super().__init__()
        self.param = param
        self.model = load_pointcloud_grasp_diffusion(param)
        loss = losses.get_losses({"Losses":['projected_denoising_loss']})
        self.loss_fn = loss.loss_fn
        self.val_loss_fn = loss.loss_fn
    
    def configure_optimizers(self):
        lr_schedules = get_learning_rate_schedules(self.param)
        optimizer = torch.optim.Adam([
            {
                "params": self.model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": self.model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": self.model.decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])
        return optimizer

    def training_step(self, batch):
        model_input = {
            "x_ene_pos":batch['pose_grasp'],
            "visual_context":batch['pcd']
        }
        losses, iter_info = self.loss_fn(self.model, model_input, None)
        loss = losses['Score loss']
        self.log("total_loss", loss, prog_bar=True)
        return loss