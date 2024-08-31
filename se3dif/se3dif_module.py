import lightning as L
from se3dif.models.loader import *
from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules
import se3dif.losses as losses
from icra2025 import ROOT
from omegaconf import OmegaConf
from SPLIT.metric import calculate_emd
from se3dif.samplers import Grasp_AnnealedLD
import copy

to_tensor = lambda x, device: torch.from_numpy(x).float().to(device)
to_numpy = lambda x: x.detach().cpu().numpy()

class SE3DifModule(L.LightningModule):
    def __init__(self, param_path:Path, **kwargs):
        super().__init__()
        #spec_file = ROOT / "external/grasp_diffusion/params.json"
        self.save_hyperparameters()
        args = load_experiment_specifications(ROOT/param_path)
        args['device'] = 'cuda'
        
        args.update(kwargs)
        self.model = load_pointcloud_grasp_diffusion(args)
        loss = losses.get_losses({"Losses":['projected_denoising_loss']})
        self.loss_fn = loss.loss_fn
        self.val_loss_fn = loss.loss_fn
        self.args = args
        self.sampler = Grasp_AnnealedLD(
            self.model, device='cuda', batch=100)
        self.ws_size = self.args['ws_size']
        self.ws_center = self.args['ws_center']
    
    def configure_optimizers(self):
        lr_schedules = get_learning_rate_schedules(self.args)
        optimizer = torch.optim.Adam( #self.model.parameters(), lr=1e-4)
            [
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

    def apply_scaling(self, batch):
        if 'visual_context' in batch:
            pcd = batch['visual_context']
        elif 'pcd' in batch:
            pcd = batch['pcd']
        self._center = torch.mean(pcd, dim=1, keepdim=True)
        scale = 2. / self.ws_size
        batch['visual_context'] = (pcd - self._center) * scale
        
        if 'x_ene_pos' in batch:
            pose = batch['x_ene_pos']
            no_pose = torch.einsum("bnij->bn", pose) == 0.
            pose[..., :3, -1] = (pose[..., :3, -1] - self._center) * scale
            pose[no_pose] = 0.
            batch['x_ene_pos'] = pose
        
        return batch
    
    def apply_reverse_scaling_to_pose(self, pose):
        pose = pose.clone()            
        
        normalized_grid_size = 2.
        #ws_center = torch.tensor(self.args['ws_center']).float().to(self.device)
        scale = normalized_grid_size / self.args['ws_size']
        
        pos = pose[..., :3, 3] / scale
        if hasattr(self, "_center"):
            pos += self._center
        pose[..., :3, 3] = pos
        return pose
    
    def training_step(self, batch):
        model_input, gt = batch
        model_input = self.apply_scaling(model_input)
        losses, iter_info = self.loss_fn(self.model, model_input, gt)
        
        train_loss = 0.
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss
        
        self.log("total_loss", train_loss, prog_bar=True)
        return loss

    def sample(self, num_samples=100):
        pose = self.sampler.sample(batch=num_samples)
        return pose.reshape(self.B, -1, 4, 4)
    
    def generate_grasp_poses(self, vision_inputs, num_samples=100):
        vision_inputs = copy.deepcopy(vision_inputs)
        vision_inputs['pcd'] = to_tensor(vision_inputs['pcd'], self.device).unsqueeze(0)
        vision_inputs = self.apply_scaling(vision_inputs)
        self.B = 1
        self.model.set_latent(
            vision_inputs['visual_context'], 
            batch=num_samples)
        grasp_poses = self.sample(num_samples)
        grasp_poses_w = self.apply_reverse_scaling_to_pose(grasp_poses)
        return grasp_poses_w[0]

    def validation_step(self, batch):
        model_input, gt = batch
        model_input = self.apply_scaling(model_input)
        
        #encode shape
        c = model_input['visual_context']
        B, P = c.shape[0], 100
        self.model.set_latent(c, batch=P)

        poses_gt = model_input['x_ene_pos']
        poses_pred = self.sample(num_samples=B*P).reshape(B, P, 4, 4)
        
        emds = []
        for pose_gt, pose_pred in zip(poses_gt, poses_pred):
            if pose_gt.sum() == 0: continue
            emds += [calculate_emd(pose_gt, pose_pred).item()]
        
        self.log("total_emd", np.mean(emds))
            
# from se3dif.utils import load_experiment_specifications
# import torch
# from torch.utils.data import DataLoader
# from se3dif.datasets.acronym_dataset import PointcloudDataset as PointcloudDataset
# from icra2025 import ROOT
# from se3dif.models import loader
# from se3dif import losses, summaries   #, trainer

# if __name__ == "__main__":
#     spec_file = ROOT / "external/grasp_diffusion/params.json"
#     args = load_experiment_specifications(spec_file)
    
    # train_dataset = PointcloudDataset(
    #     "data/scene_grasp_data/floating_mbbchl/train/*.h5",
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=8,
    #     persistent_workers=True)
    
    # args['device'] = 'cuda'
    # model = loader.load_model(args)
    # loss = losses.get_losses(args)
    # loss_fn = val_loss_fn = loss.loss_fn
    # summary = summaries.get_summary(args, False)

    ## Optimizer
    # lr_schedules = get_learning_rate_schedules(args)
    # optimizer = torch.optim.Adam([
    #     {
    #         "params": model.vision_encoder.parameters(),
    #         "lr": lr_schedules[0].get_learning_rate(0),
    #     },
    #     {
    #         "params": model.feature_encoder.parameters(),
    #         "lr": lr_schedules[1].get_learning_rate(0),
    #     },
    #     {
    #         "params": model.decoder.parameters(),
    #         "lr": lr_schedules[2].get_learning_rate(0),
    #     },
    # ])
    
    # module = SE3DifModule(args)
    # tr = L.Trainer(
    #     default_root_dir=ROOT/"se3dif_models",
    #     max_epochs=10
    # )
    # tr.fit(
    #     module, train_dataloader
    # )
    # trainer.train(
    #     model=model.float(), 
    #     train_dataloader=train_dataloader, 
    #     epochs=args['TrainSpecs']['num_epochs'],
    #     model_dir= ROOT/"se3dif_models",
    #     summary_fn=summary, device="cuda", lr=1e-4, optimizers=[optimizer],
    #     steps_til_summary=args['TrainSpecs']['steps_til_summary'],
    #     epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
    #     loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
    #     clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
    # )
    