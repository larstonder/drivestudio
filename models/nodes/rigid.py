from typing import Dict, List, Tuple
import logging
import random

import torch
from torch.nn import Parameter

from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians

logger = logging.getLogger()

class RigidNodes(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        print(f"Initializing RigidNodes with {kwargs}")
        self.dataset_id_to_model_id_map: Dict[int, int] = {}
        self.model_id_to_dataset_id_map: Dict[int, int] = {}
        self.track_token_to_model_id_map: Dict[str, int] = {}
        self.model_id_to_track_token_map: Dict[int, str] = {}
        
    @property
    def num_instances(self):
        return self.instances_fv.shape[1]
    @property
    def num_frames(self):
        return self.instances_fv.shape[0]
    
    def get_pts_valid_mask(self):
        """
        get the mask for valid points
        """
        return self.instances_fv[self.cur_frame][self.point_ids[..., 0]]
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        
    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        instance_pts_dict: {
            id in dataset: { # Note: This key is usually a string from json, but represents an int
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
                "track_token": str or None,
            },
        }
        """
        
        print(f"Creating RigidNodes from pcd with {len(instance_pts_dict)} instances")
        # collect all instances
        init_means = []
        init_colors = []
        instances_pose = []
        instances_size = []
        instances_fv = []
        point_ids = []
        
        current_model_id = 0
        temp_dataset_id_to_model_id_map = {}
        temp_model_id_to_dataset_id_map = {}
        temp_track_token_to_model_id_map = {}
        temp_model_id_to_track_token_map = {}

        for id_in_dataset_str, v in instance_pts_dict.items():
            # Get the actual track token from the data
            track_token = v.get("track_token", None)
            
            try:
                # Ensure id_in_dataset is an integer, as it comes from JSON string keys
                id_in_dataset = int(id_in_dataset_str)
            except ValueError:
                print(f"Could not convert dataset ID '{id_in_dataset_str}' to int. Skipping this instance.")
                continue

            init_means.append(v["pts"])
            init_colors.append(v["colors"])
            instances_pose.append(v["poses"].unsqueeze(1))
            instances_size.append(v["size"])
            instances_fv.append(v["frame_info"].unsqueeze(1))
            point_ids.append(torch.full((v["num_pts"], 1), current_model_id, dtype=torch.long))
            
            temp_dataset_id_to_model_id_map[id_in_dataset] = current_model_id
            temp_model_id_to_dataset_id_map[current_model_id] = id_in_dataset
            
            # Add the track token mapping only if track_token is not None
            if track_token is not None:
                temp_track_token_to_model_id_map[track_token] = current_model_id
                temp_model_id_to_track_token_map[current_model_id] = track_token
                
            current_model_id += 1
            
        self.dataset_id_to_model_id_map = temp_dataset_id_to_model_id_map
        self.model_id_to_dataset_id_map = temp_model_id_to_dataset_id_map
        self.track_token_to_model_id_map = temp_track_token_to_model_id_map
        self.model_id_to_track_token_map = temp_model_id_to_track_token_map

        # Debug: Print the mapping information
        logger.info(f"RigidNodes initialized with {len(self.track_token_to_model_id_map)} track tokens:")
        for track_token, model_id in list(self.track_token_to_model_id_map.items())[:5]:  # Show first 5
            print(f"  Track token {track_token} -> Model ID {model_id}")
        if len(self.track_token_to_model_id_map) > 5:
            print(f"  ... and {len(self.track_token_to_model_id_map) - 5} more")

        init_means = torch.cat(init_means, dim=0).to(self.device) # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device) # (N, 3)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device) # (num_frame, num_instances, 4, 4)
        self.instances_size = torch.stack(instances_size).to(self.device) # (num_instances, 3)
        self.instances_fv = torch.cat(instances_fv, dim=1).to(self.device) # (num_frame, num_instances)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)
        instances_quats = self.get_instances_quats(instances_pose)
        instances_trans = instances_pose[..., :3, 3]
        
        # initialize the means, scales, quats, and colors
        self._means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)
        
        # pose refinement
        self.instances_quats = Parameter(self.quat_act(instances_quats)) # (num_frame, num_instances, 4)
        self.instances_trans = Parameter(instances_trans)              # (num_frame, num_instances, 3)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix+"ins_translation"] = [self.instances_trans]
        return param_groups

    def get_instances_quats(self, instances_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert the pose to quaternion for all frames and instances
        """
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames*num_instances, 4, device=self.device)
        
        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = self.instances_fv.view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)
        
        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_ids

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_ids

    def get_out_of_bound_mask(self):
        """
        This function checks if the gaussians are out of instance boxes
        """
        # get the instance boxes
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means
        
        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            # use the previous and next frame to interpolate the pose
            _quats_prev_frame = self.instances_quats[self.cur_frame - 1]
            _quats_next_frame = self.instances_quats[self.cur_frame + 1]
            _quats_cur_frame = self.instances_quats[self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            quats_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_quats, _quats_cur_frame
            )
        else:
            quats_cur_frame = self.instances_quats[self.cur_frame] # (num_instances, 4)
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )                                                          # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]        # (num_points, 3, 3)
        
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means = torch.bmm(
            rot_per_pts, means.unsqueeze(-1)
        ).squeeze(-1) + trans_per_pts
        return means

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        """
        transform the quats of instances to world space
        according to the pose at the current frame
        """
        assert quats.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        global_quats_cur_frame = self.instances_quats[self.cur_frame]
        global_quats_per_pts = global_quats_cur_frame[self.point_ids[..., 0]]
            
        global_quats_per_pts = self.quat_act(global_quats_per_pts)
        _quats = self.quat_act(quats)
        return quat_mult(global_quats_per_pts, _quats)

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        # NOTE: hack here, need to consider a gaussian filter for efficient rendering
        
        world_means = self.transform_means(self._means)
        world_quats = self.transform_quats(self._quats)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_scales = self.get_scaling
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
        
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(self._scales[pts_mask])
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,            
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict
    
    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        scaling_reg = self.reg_cfg.get("scaling_reg", None)
        if scaling_reg is not None:
            w = scaling_reg.w
            precentile = scaling_reg.precentile
            stop_after = scaling_reg.stop_after
            start_after = scaling_reg.start_after
            
            if self.step < stop_after and self.step > start_after and w > 0:
                scale_prod = self._gs_cache["_scales"].prod(dim=-1)
                p = torch.kthvalue(scale_prod, int(scale_prod.shape[0] * precentile)).values
                # penalize the scales that are too large
                loss_dict["scaling_percentile_reg"] = torch.relu(scale_prod - p).mean() * w

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            instance_mask = self.instances_fv[self.cur_frame]
            if instance_mask.sum() > 0:
                trans_cfg = temporal_smooth_reg.get("trans", None)
                if trans_cfg is not None:
                    fi_interval = random.randint(1, trans_cfg.smooth_range)
                    if self.cur_frame >= fi_interval and self.cur_frame < self.num_frames - fi_interval:
                        valid_mask = (
                            self.instances_fv[self.cur_frame - fi_interval] & \
                            self.instances_fv[self.cur_frame + fi_interval] & \
                            self.instances_fv[self.cur_frame]
                        )
                        if valid_mask.sum() > 0:
                            cur_trans = self.instances_trans[self.cur_frame]
                            pre_trans = self.instances_trans[self.cur_frame - fi_interval].data
                            next_trans = self.instances_trans[self.cur_frame + fi_interval].data
                            loss = (next_trans[valid_mask] + pre_trans[valid_mask] - 2 * cur_trans[valid_mask]).abs().mean()
                            loss_dict["trans_temporal_smooth"] = loss * trans_cfg.w
        return loss_dict

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update({
            "points_ids": self.point_ids,
            "instances_size": self.instances_size,
            "instances_fv": self.instances_fv,
            # Add the mapping dictionaries
            "dataset_id_to_model_id_map": self.dataset_id_to_model_id_map,
            "model_id_to_dataset_id_map": self.model_id_to_dataset_id_map,
            "track_token_to_model_id_map": self.track_token_to_model_id_map,
            "model_id_to_track_token_map": self.model_id_to_track_token_map,
        })
        return state_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        
        # Restore the mapping dictionaries, providing default empty dicts for backward compatibility
        self.dataset_id_to_model_id_map = state_dict.pop("dataset_id_to_model_id_map", {})
        self.model_id_to_dataset_id_map = state_dict.pop("model_id_to_dataset_id_map", {})
        self.track_token_to_model_id_map = state_dict.pop("track_token_to_model_id_map", {})
        self.model_id_to_track_token_map = state_dict.pop("model_id_to_track_token_map", {})
        
        self.instances_trans = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
        )
        self.instances_quats = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 4, device=self.device)
        )
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    # editting functions
    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        remove instances from the model
        
        Args:
            remove_id_list: list of instance ids to be removed
        """
        for ins_ids in remove_id_list:
            mask = ~(self.point_ids[..., 0] == ins_ids)
            self._means = Parameter(self._means[mask])
            self._scales = Parameter(self._scales[mask])
            self._quats = Parameter(self._quats[mask])
            self._features_dc = Parameter(self._features_dc[mask])
            self._features_rest = Parameter(self._features_rest[mask])
            self._opacities = Parameter(self._opacities[mask])
            self.point_ids = self.point_ids[mask]
        
    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                instance_raw_dict = {
                    "_means": self._means[self.point_ids[..., 0] == id],
                    "_scales": self._scales[self.point_ids[..., 0] == id],
                    "_quats": self._quats[self.point_ids[..., 0] == id],
                    "_features_dc": self._features_dc[self.point_ids[..., 0] == id],
                    "_features_rest": self._features_rest[self.point_ids[..., 0] == id],
                    "_opacities": self._opacities[self.point_ids[..., 0] == id],
                    "point_ids": self.point_ids[self.point_ids[..., 0] == id],
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0))
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0))
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)
    
    def translate_instance(self, ins_id: int, delta_xyz: torch.Tensor) -> None:
        """
        translate instance with id ins_id
        
        Args:
            ins_id: instance id
            delta_xyz: translation vector
        """
        # Find points belonging to this instance
        pts_mask = self.point_ids[..., 0] == ins_id
        
        # Apply translation directly to the means
        self._means[pts_mask] = self._means[pts_mask] + delta_xyz
    
    def rotate_instance(self, ins_id: int, delta_quat: torch.Tensor) -> None:
        """
        rotate instance with id ins_id
        
        Args:
            ins_id: instance id
            delta_quat: rotation quaternion
        """
        pts_mask = self.point_ids[..., 0] == ins_id
        # rotate the points
        means = self._means[pts_mask]
        delta_quat = self.quat_act(delta_quat)  # normalize quaternion
        delta_rot = quat_to_rotmat(delta_quat)  # convert to rotation matrix
        
        # Apply rotation around the instance center
        center = means.mean(dim=0, keepdim=True)
        centered_points = means - center
        rotated_points = torch.bmm(delta_rot.unsqueeze(0).expand(centered_points.shape[0], -1, -1), 
                                   centered_points.unsqueeze(-1)).squeeze(-1)
        self._means[pts_mask] = rotated_points + center
        
        # Also update the quaternions of the gaussians
        quats = self._quats[pts_mask]
        normalized_quats = self.quat_act(quats)
        # Multiply quaternions: new_q = delta_q * old_q
        self._quats[pts_mask] = self.quat_act(quat_mult(delta_quat.unsqueeze(0).expand_as(normalized_quats), normalized_quats))
    
    def scale_instance(self, ins_id: int, delta_scale: torch.Tensor) -> None:
        """
        scale instance with id ins_id
        
        Args:
            ins_id: instance id
            delta_scale: scaling vector (log scale)
        """
        pts_mask = self.point_ids[..., 0] == ins_id
        
        # Get current means and center point
        means = self._means[pts_mask]
        center = means.mean(dim=0, keepdim=True)
        
        # Apply scaling from the center point
        centered_points = means - center
        
        # Apply delta_scale to the log scale values
        self._scales[pts_mask] = self._scales[pts_mask] + delta_scale
        
        # Also scale the points' positions from center
        # Convert delta_scale from log space to linear space
        linear_scale = torch.exp(delta_scale)
        scaled_points = centered_points * linear_scale
        self._means[pts_mask] = scaled_points + center

    def export_gaussians_to_ply(self, alpha_thresh: float, instance_id: List[int] = None) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == instance_id
        
        means = self._means[pts_mask]
        direct_color = self.colors[pts_mask]
        
        activated_opacities = self.get_opacity[pts_mask]
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }