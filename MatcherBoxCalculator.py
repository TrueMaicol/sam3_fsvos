from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class MatcherBoxCalculator():
    def __init__(self, sam3_model=None, sam3_processor=None):
        if sam3_model is None:
            raise ValueError("sam3 model must be specified")

        self.model = sam3_model
        self.processor = sam3_processor
        self.resolution = 1008 # hardcoded from SAM3
        self.input_size = (self.resolution, self.resolution)
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.resolution, self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_patch_size = 14
        # Note: 1008 / 14 = 72. 256 is the embedding dimension (d_model), 
        # but for spatial matching logic we use the grid size (72).
        self.encoder_feat_size = 72 
        
        # Flags inferred from Matcher core
        self.use_box = True
        self.use_negative_priors_from_discarded = False
        self.use_negative_priors_from_cost = False
        
        # Matcher internal state initialization
        self.ref_masks_pool = None
        self.S = None
        self.S_forward = None
        self.S_reverse = None
        self.sim_scores_after_forward_matching = None
        self.sim_scores_after_backward_matching = None
        self.sim_discarded_patches = None
        self.number_support_patches_forward_matching = None
        self.number_query_patches_forward_matching = None
        self.number_support_patches_backward_matching = None
        self.number_query_patches_backward_matching = None
    
    def get_image_features(self, image):
        image = v2.functional.to_image(image)
        
        image = image.to(self.device).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = self.transform(image)

        with torch.no_grad():
            backbone_out = self.model.backbone.forward_image(image)
            # Use scale 1.0 features (72x72) to match patch_size 14
            visual_features = backbone_out["backbone_fpn"][2] # [1, 256, 72, 72]
            print(f"[MatcherBoxCalculator] Backbone Output Shape: {visual_features.shape}")
            
        b, c, h, w = visual_features.shape
        # Cast to float32: the backbone produces bfloat16 (model is bfloat16),
        # but matcher operations (scipy/numpy) require float32.
        feats = visual_features.float().view(b, c, -1).permute(0, 2, 1).reshape(-1, c)
        print(f"[MatcherBoxCalculator] Flattened Features Shape: {feats.shape}")
        feats = F.normalize(feats, dim=1, p=2)
        
        return feats

    def get_fused_image_features(self, image, text_prompt="visual", skip_coords=False):
        if self.processor is None:
            raise ValueError("sam3_processor must be provided to use fused features")
            
        if image.ndim == 3:
            image = image.unsqueeze(0)

        all_feats = []
        for img in image:
            state = self.processor.set_image(img)
            text_outputs = self.processor.model.backbone.forward_text([text_prompt], device=self.processor.device)
            state["backbone_out"].update(text_outputs)
            state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
            
            prompt, prompt_mask, backbone_out = self.processor.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                encode_text=True,
                skip_coords=skip_coords
            )
            
            backbone_out, encoder_out, _ = self.processor.model._run_encoder(
                backbone_out, self.processor.find_stage, prompt, prompt_mask
            )
            
            feat = encoder_out["encoder_hidden_states"].squeeze(1).float()
            feat = F.normalize(feat, dim=1, p=2)
            all_feats.append(feat)

        out_feats = torch.cat(all_feats, dim=0)
        return out_feats

    def compute_box(self, reference_image=None, target_image=None, reference_mask=None, text_prompt="visual", use_fused_matcher_features=False, skip_coords=False):
        if reference_image is None or target_image is None:
            raise ValueError("Reference or Target image is not specified")
        

        if reference_mask is None:
            reference_mask = torch.ones((self.resolution, self.resolution), device=self.device)
        elif not isinstance(reference_mask, torch.Tensor):
            reference_mask = torch.from_numpy(np.array(reference_mask)).to(self.device).float()
        
        if reference_mask.ndim == 2:
            reference_mask = reference_mask.unsqueeze(0).unsqueeze(0)
        elif reference_mask.ndim == 3:
            reference_mask = reference_mask.unsqueeze(0)
            
        if reference_mask.shape[-2:] != (self.resolution, self.resolution):
            reference_mask = F.interpolate(reference_mask, size=(self.resolution, self.resolution), mode='nearest')
        print(f"[MatcherBoxCalculator] Reference Mask Shape: {reference_mask.shape}")

        target_image = target_image.to(self.device)
        reference_image = reference_image.to(self.device)
        reference_mask = reference_mask.to(self.device)

        # Pool mask to 72x72
        ref_masks_pool_grid = F.avg_pool2d(reference_mask, (self.encoder_patch_size, self.encoder_patch_size))
        print(f"[MatcherBoxCalculator] Pooled Mask Grid Shape: {ref_masks_pool_grid.shape}")
        self.ref_masks_pool = (ref_masks_pool_grid > 0.01).float().reshape(-1)

        if use_fused_matcher_features:
            ref_features = self.get_fused_image_features(reference_image, text_prompt, skip_coords)
            target_features = self.get_fused_image_features(target_image, text_prompt, skip_coords)
        else:
            ref_features = self.get_image_features(reference_image)
            target_features = self.get_image_features(target_image)

        # Returns: ponits, negative_priors if len(negative_priors) > 0 else points_discarded, box, self.S, C, reduced_points_num, reduced_points_num_neg, matched_features, target_features, matched_indices_in_all
        results = self.patch_level_matching(ref_features, target_features)
        box = results[2]
        points = results[0]
        matched_features = results[7]
        all_target_features = results[8]
        matched_indices_in_all = results[9]
        return box, points, matched_features, all_target_features, matched_indices_in_all

    def patch_level_matching(self, ref_feats, tar_feat):
        """
        Performs patch-level matching between the reference and target image features.
        """
        self.tar_feat = tar_feat
        # forward matching
        self.S = ref_feats @ tar_feat.t()  # S = ns*N x N

        # from cosine similarity ----> to cosine distance C = ns*N x N
        C = (1 - self.S) / 2

        # keeping only the points of the reference/support image that are within the support mask.
        self.S_forward = self.S[self.ref_masks_pool.flatten().bool()]
        number_support_patches = self.S_forward.shape[0]
        # S_forward = T x N, where T is the number of points within the mask.

        # Patches in the reference/support image feature(s) and the target image features are seen as nodes in a bipartite graph.
        # The similarity matrix S is used to compute the optimal matching between the two sets of nodes.
        indices_forward = linear_sum_assignment(
            self.S_forward.float().cpu(), maximize=True)

        # Indices forward will contain 2 tuples: the first tuple will contain the indices of the reference patches, the second tuple will contain the
        # indices of the target patches that have been matched. We first convert them to tensors and then from the similarity matrix S_forward we extract
        # the similarity scores of the matched patches.
        indices_forward = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_forward]
        self.number_support_patches_forward_matching = len(indices_forward[0])
        self.number_query_patches_forward_matching = len(indices_forward[1])
        # sim_scores_f = T, i.e. the similarity scores of the matched patches.
        sim_scores_f = self.S_forward[indices_forward[0], indices_forward[1]]
        self.sim_scores_after_forward_matching = sim_scores_f
        # self.ref_masks_pool.flatten() = ns*N, self.ref_masks_pool.flatten().nonzero() = T x 1,
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]
        # self.ref_masks_pool.flatten().nonzero()[:, 0] = T, i.e. the indices of the patches within the mask.

        # reverse matching
        # S.t() = N x ns*N, S_reverse = K x ns*N. We are keeping only the points of the target image that have been matched.
        self.S_reverse = self.S.t()[indices_forward[1]]

        # indices_reverse will contain 2 tuples: the first tuple will contain the indices of the target patches,
        indices_reverse = linear_sum_assignment(
            self.S_reverse.float().cpu(), maximize=True)
        # the second tuple will contain the indices of the reference patches that have been matched.
        indices_reverse = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        # I want to retain only the indices of the reference/support patches that have been matched in the reverse matching and
        retain_ind = torch.isin(indices_reverse[1], indices_mask)

        # that are within the initial reference/support mask.
        indices_forward_pos = indices_forward
        indices_forward_neg = indices_forward
        sim_scores_f_pos = sim_scores_f.clone()
        sim_scores_f_neg = sim_scores_f.clone()
        self.number_support_patches_backward_matching = len(indices_forward[0])
        self.number_query_patches_backward_matching = len(indices_forward[1])
        if not (retain_ind == False).all().item():
            indices_forward_pos = [indices_forward[0]
                                   [retain_ind], indices_forward[1][retain_ind]]
            indices_forward_neg = [indices_forward[0]
                                   [~retain_ind], indices_forward[1][~retain_ind]]
            sim_scores_f_pos = sim_scores_f[retain_ind]
            sim_scores_f_neg = sim_scores_f[~retain_ind]
            self.number_support_patches_backward_matching = len(
                indices_forward[0][retain_ind])
            self.number_query_patches_backward_matching = len(
                indices_forward[1][retain_ind])
        else:
            print('[WARNING] - All the matched points have been discarded.')

        inds_matched, sim_matched = indices_forward_pos, sim_scores_f_pos
        self.sim_scores_after_backward_matching = sim_matched
        self.sim_discarded_patches = sim_scores_f_neg

        # if there are more than 40 matched points, we keep only half of them.
        reduced_points_num = len(
            sim_matched) // 2 if len(sim_matched) > 40 else len(sim_matched)
        sim_sorted, sim_idx_sorted = torch.sort(sim_matched, descending=True)
        sim_filter = sim_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward_pos[1][sim_filter]
        points_unmatched_inds = indices_forward_neg[1]

        # removing duplicates while preserving the similarity order - (Deterministic Baseline)
        unique_inds = []
        seen = set()
        for ind in points_matched_inds.cpu().tolist():
            if ind not in seen:
                unique_inds.append(ind)
                seen.add(ind)
        points_matched_inds_set = torch.tensor(unique_inds, device=self.device)
        
        # get the features of the matched points
        matched_features = self.tar_feat[points_matched_inds_set]

        # for negative priors (unmatched)
        unique_inds_neg = []
        seen_neg = set()
        for ind in points_unmatched_inds.cpu().tolist():
            if ind not in seen_neg:
                unique_inds_neg.append(ind)
                seen_neg.add(ind)
        points_unmatched_inds_set = torch.tensor(unique_inds_neg, device=self.device)

        # getting the x coordinate of the matched points
        points_matched_inds_set_w = points_matched_inds_set % self.encoder_feat_size
        points_unmatched_inds_set_w = points_unmatched_inds_set % self.encoder_feat_size
        # getting the y coordinate of the matched points
        points_matched_inds_set_h = points_matched_inds_set // self.encoder_feat_size
        points_unmatched_inds_set_h = points_unmatched_inds_set // self.encoder_feat_size

        # converting the x coordinate to the original image coordinate
        idxs_mask_set_x = (points_matched_inds_set_w *
                           self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
        idxs_mask_set_x_unmatched = (
            points_unmatched_inds_set_w * self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
        # converting the y coordinate to the original image coordinate
        idxs_mask_set_y = (points_matched_inds_set_h *
                           self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
        idxs_mask_set_y_unmatched = (
            points_unmatched_inds_set_h * self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
        # Note that the original image coordinate is the coordinate of the center of the patch.

        ponits_matched = []
        points_discarded = []
        keep_indices = []
        # retaining only the points that are within the image shape
        for i, (x, y) in enumerate(zip(idxs_mask_set_x, idxs_mask_set_y)):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                ponits_matched.append([int(x), int(y)])
                keep_indices.append(i)

        for x, y in zip(idxs_mask_set_x_unmatched, idxs_mask_set_y_unmatched):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_discarded.append([int(x), int(y)])

        ponits = np.array(ponits_matched)
        matched_features = matched_features[keep_indices]
        points_discarded = np.array(points_discarded)

        # Sampling negative points from the discarded points and from the cost matrix.
        # The negative points are used as negative priors in the mask generation.
        negative_priors = []
        reduced_points_num_neg = []
        if self.use_negative_priors_from_discarded:
            negative_priors_from_discarded, reduced_points_num_neg_from_discarded = self.sample_negative_points_from_discarded(
                indices_forward, sim_scores_f, indices_reverse, indices_mask)
            negative_priors.append(negative_priors_from_discarded)
            reduced_points_num_neg.append(
                reduced_points_num_neg_from_discarded)
        if self.use_negative_priors_from_cost:
            negative_priors_from_cost, reduced_points_num_neg_from_cost = self.sample_negative_points_from_cost(
                C)
            negative_priors.append(negative_priors_from_cost)
            reduced_points_num_neg.append(reduced_points_num_neg_from_cost)

        # In case of bounding box prompts are added, the box is computed as the bounding box of the matched points.
        if self.use_box:
            box = np.array([
                max(ponits[:, 0].min(), 0),
                max(ponits[:, 1].min(), 0),
                min(ponits[:, 0].max(), self.input_size[1] - 1),
                min(ponits[:, 1].max(), self.input_size[0] - 1),
            ])
        else:
            box = None

        return ponits, negative_priors if len(negative_priors) > 0 else points_discarded, box, self.S, C, reduced_points_num, reduced_points_num_neg, matched_features, self.tar_feat, points_matched_inds_set[keep_indices]

    def sample_negative_points_from_discarded(self, idxs_forward, sim_scores_forward, idxs_reverse, idxs_mask):
        # I want to retain the indices that are not matched to use them as negative priors.
        discarded_ind = torch.isin(idxs_reverse[1], idxs_mask, invert=True)

        indices_forward_neg = None
        dissim_scores_f = None
        if not (discarded_ind == False).all().item():
            indices_forward_neg = [
                idxs_forward[0][discarded_ind], idxs_forward[1][discarded_ind]]
            dissim_scores_f = sim_scores_forward[discarded_ind]
        inds_unmatched, sim_unmatched = indices_forward_neg, dissim_scores_f

        if indices_forward_neg is not None:
            reduced_points_num_neg = len(
                sim_unmatched) // 2 if len(sim_unmatched) > 40 else len(sim_unmatched)
            sim_sorted_neg, sim_idx_sorted_neg = torch.sort(
                sim_unmatched, descending=False)
            sim_filter_neg = sim_idx_sorted_neg[:reduced_points_num_neg]
            # These are the candidates negative priors
            points_unmatched_inds = indices_forward_neg[1][sim_filter_neg]
        else:
            reduced_points_num_neg = None
            points_unmatched_inds = None

        # Handling the negative priors
        if points_unmatched_inds is not None:
            points_unmatched_inds_set = torch.tensor(
                list(set(points_unmatched_inds.cpu().tolist())))  # removing duplicates from negative priors
            points_unmatched_inds_set_w = points_unmatched_inds_set % (
                self.encoder_feat_size)  # getting the x coordinate of the unmatched points
            points_unmatched_inds_set_h = points_unmatched_inds_set // (
                self.encoder_feat_size)
            idxs_mask_set_x_neg = (points_unmatched_inds_set_w *
                                   self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
            idxs_mask_set_y_neg = (points_unmatched_inds_set_h *
                                   self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
            points_unmatched = []
            for x, y in zip(idxs_mask_set_x_neg, idxs_mask_set_y_neg):
                if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                    points_unmatched.append([int(x), int(y)])
            negative_priors = np.array(points_unmatched)
        else:
            negative_priors = None

        return negative_priors, reduced_points_num_neg

    def sample_negative_points_from_cost(self, C):
        C_forward = C.clone()

        # Perform forward matching to find the most similar patches in the target image for each patch in the reference image.
        indices_forward_neg = linear_sum_assignment(
            C_forward.float().cpu(), maximize=True)
        indices_forward_neg = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_forward_neg]
        cost_scores_forward = C_forward[indices_forward_neg[0],
                                        indices_forward_neg[1]]

        # Get the indices of the patches within the reference image mask.
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]

        # Perform reverse matching to find the patches in the reference image that are most similar to each patch in the target image.
        C_reverse = C.t()[indices_forward_neg[1]]
        indices_reverse = linear_sum_assignment(C_reverse.float().cpu(), maximize=True)
        indices_reverse = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        retain_ind = torch.isin(indices_reverse[1], indices_mask, invert=True)

        # Keep only the negative points that are not within the reference image mask.
        indices_forward_neg_f = indices_forward_neg
        cost_scores_f = cost_scores_forward.clone()
        if not (retain_ind == False).all().item():
            indices_forward_neg = [
                indices_forward_neg_f[0][retain_ind], indices_forward_neg_f[1][retain_ind]]
            cost_scores_f = cost_scores_f[retain_ind]
        inds_neg_matched, cost_matched = indices_forward_neg_f, cost_scores_f

        # If there are more than 40 matched points, keep only half of them.
        reduced_points_num = len(
            cost_matched) // 2 if len(cost_matched) > 40 else len(cost_matched)
        cost_sorted, cost_idx_sorted = torch.sort(
            cost_matched, descending=True)
        cost_filter = cost_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward_neg_f[1][cost_filter]

        # Remove duplicate points and convert the indices to the original image coordinates.
        points_matched_inds_set = torch.tensor(
            list(set(points_matched_inds.cpu().tolist())))
        points_matched_inds_set_w = points_matched_inds_set % (
            self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (
            self.encoder_feat_size)
        idxs_mask_set_x = (points_matched_inds_set_w *
                           self.encoder_patch_size + self.encoder_patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h *
                           self.encoder_patch_size + self.encoder_patch_size // 2).tolist()

        # Create the array of negative points in the original image coordinates.
        points_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_matched.append([int(x), int(y)])
        points = np.array(points_matched)

        return points, reduced_points_num

    def convert_box_to_input_resolution(self, box=None, output_resolution=518):
        if box is None:
            raise Exception("Box is not specified")
        
        assert isinstance(output_resolution, (int, tuple, list))
        if isinstance(output_resolution, int):
            out_resolution = (output_resolution, output_resolution)
        else:
            out_resolution = output_resolution

        x1, y1, x2, y2 = box
        fin_W, fin_H = out_resolution

        rescale_X = fin_W / self.resolution
        rescale_Y = fin_H / self.resolution

        out_box = (x1 * rescale_X, y1 * rescale_Y, x2 * rescale_X, y2 * rescale_Y)
        
        return out_box
        
    def convert_points_to_input_resolution(self, points=None, output_resolution=518):
        if points is None:
            raise Exception("Points are not specified")
        
        assert isinstance(output_resolution, (int, tuple, list))
        if isinstance(output_resolution, int):
            out_resolution = (output_resolution, output_resolution)
        else:
            out_resolution = output_resolution

        rescale_X = out_resolution[0] / self.resolution
        rescale_Y = out_resolution[1] / self.resolution

        out_points = []
        for x, y in points:
            out_x = x * rescale_X
            out_y = y * rescale_Y
            out_points.append((out_x, out_y))
        
        return np.array(out_points)
