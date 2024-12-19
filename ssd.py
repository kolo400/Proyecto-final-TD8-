import torch
import torch.nn.functional as F
import warnings
import torchvision

from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import vgg, resnet
from torch.hub import load_state_dict_from_url
from torchvision.ops import boxes as box_ops

__all__ = ['SSD', 'ssd300_vgg16', 'ssd512_resnet50']

model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
    'ssd512_resnet50_coco': 'https://download.pytorch.org/models/ssd512_resnet50_coco-d6d7edbb.pth',
}

backbone_urls = {
    # We port the features of a VGG16 backbone trained by amdegroot because unlike the one on TorchVision, it uses the
    # same input standardization method as the paper. Ref: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
    'vgg16_features': 'https://download.pytorch.org/models/vgg16_features-amdegroot.pth'
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'bbox_regression': self.regression_head(x),
            'cls_logits': self.classification_head(x),
        }


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        #print(self.module_list)
        for module in self.module_list:
            if i == idx:
                #print(x.shape)
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            #print('shape loco:')
            #print(results.shape)
            #print(self.num_columns)
            results = results.view(N, -1, self.num_columns, H, W)
            #print('A mi no se me rompio loco')
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            #print(channels,anchors)
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSD(nn.Module):
    """
    Implements SSD architecture from `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute with the list of the output channels of
            each feature map. The backbone should return a single Tensor or an OrderedDict[Tensor].
        anchor_generator (DefaultBoxGenerator): module that generates the default boxes for a
            set of feature maps.
        size (Tuple[int, int]): the width and height to which images will be rescaled before feeding them
            to the backbone.
        num_classes (int): number of output classes of the model (excluding the background).
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        head (nn.Module, optional): Module run on top of the backbone features. Defaults to a module containing
            a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        topk_candidates (int): Number of best detections to keep before NMS.
        positive_fraction (float): a number between 0 and 1 which indicates the proportion of positive
            proposals used during the training of the classification head. It is used to estimate the negative to
            positive ratio.
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator,
                 size: Tuple[int, int], num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 head: Optional[nn.Module] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25):
        super().__init__()

        self.backbone = backbone

        self.anchor_generator = anchor_generator

        self.box_coder = det_utils.BoxCoder(weights=(10., 10., 5., 5.))

        if head is None:
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = det_utils.retrieve_out_channels(backbone, size)
            
            assert len(out_channels) == len(anchor_generator.aspect_ratios)

            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head

        self.proposal_matcher = det_utils.SSDMatcher(iou_thresh)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min(size), max(size), image_mean, image_std,
                                                  size_divisible=1, fixed_size=size)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor],
                      detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs['bbox_regression'] 
        cls_logits = head_outputs['cls_logits'] 

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image,
             matched_idxs_per_image) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0] #Indices de las cajas se matchearon con una caja real 
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ))

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros((cls_logits_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = \
                targets_per_image['labels'][foreground_matched_idxs_per_image]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction='none'
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float('inf')  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            'bbox_regression': bbox_loss.sum() / N,
            'classification': (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        
        features = self.backbone(images.tensors)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])


        features = list(features.values())
        
        # compute the ssd heads outputs using the features

        
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image['boxes'].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                                   device=anchors_per_image.device))
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections


# VGG

class SSDFeatureExtractorVGG(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(
            *backbone[:maxpool4_pos]  # until conv4_3
        )

        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.ReLU(inplace=True),
            )
        ])
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            fc,
        ))
        self.extra = extra





    

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # L2 regularization + Rescaling of 1st block's feature map
    
        x = self.features(x)
        
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled] 

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])





#Dense Net

class SSDFeatureExtractorDenseNet(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        self.backbone = backbone  

        self.transition2 = nn.Sequential(
            *list(self.backbone.children())[:6]  # Incluye Dense Block 2 + Transition Layer 2
        )

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)


        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.ReLU(inplace=True),
            )
        ])
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(fc))
        self.extra = extra





    

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        
        x = self.transition2(x)
        output = [x] 

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])





#Swin Transformer

class SSDFeatureExtractorSwin(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        self.swin_backbone = backbone 
        self.stages = nn.Sequential(*self.swin_backbone[:3]) 


        
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.ReLU(inplace=True),
            )
        ])
        if highres:
            
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(
            fc, 
        ))
        self.extra = extra





    

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        
        x = self.stages(x)
        x = x.permute(0,3,2,1) 
        output = [x]

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])



#Conv Next
import torch.nn as nn
from torchvision.models import convnext_tiny

class SSDFeatureExtractorConvNeXt(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        # Obtenemos las capas jerárquicas de ConvNeXt
        self.stages = nn.ModuleList([
        nn.Sequential(*list(backbone.children())[0:2]),  # Etapa 1
        nn.Sequential(*list(backbone.children())[2:4]),  # Etapa 2
        nn.Sequential(*list(backbone.children())[4:6]),  # Etapa 3
        nn.Sequential(*list(backbone.children())[6:])   # Etapa 4
        ])

        # Capas extra para SSD
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            )
            
            
        ])
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            self.extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  # conv11_2
                nn.ReLU(inplace=True),  # Pasa de 2x2 a 1x1
            ))
        _xavier_init(self.extra)
        
        

    def forward(self, x):
        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        
        for layer in self.extra:
            
            x = layer(x)
            feature_maps.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(feature_maps)])


    



#VGG Caso Upgraded

class SSDFeatureExtractorVGG_Upgraded(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d)) 

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(
            *backbone[:maxpool4_pos]  # until conv4_3
            
        )
        

        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])

        
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=2, stride=2, padding = 1, output_padding=1) 
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)
        self.b_norm_1 = nn.BatchNorm2d(512)
        self.b_norm_2 = nn.BatchNorm2d(1024)
        self.b_norm_3 = nn.BatchNorm2d(512)
        
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  # FC6 with atrous
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        extra.insert(0, nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            fc,
        ))
        self.extra = extra


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        
        x = self.features(x) 
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        conv_4_3_features = rescaled 
        conv_4_3_features_p1 = F.max_pool2d(conv_4_3_features, kernel_size=2, stride=2) #conv4_3 pooled
        output = []
        for idx, block in enumerate(self.extra):
            if idx == 0:
                
                conv_7_features = block(x) 
                conv_7_features_d = self.deconv_1(conv_7_features) #conv_7 deconvolution
                conv_7_features_d = self.b_norm_1(conv_7_features_d) 
                
                bloque_1 = torch.cat([conv_4_3_features, conv_7_features_d], dim=1) #conv4_3 + conv_7_deconv
                
                bloque_2 = torch.cat([conv_4_3_features_p1, conv_7_features], dim=1) #conv4_3_pooled + conv_7
                bloque_2_pooled = F.max_pool2d(bloque_2, kernel_size=2, stride=2, padding=1) # bloque_2 pooled
                
                
            if idx == 1:
                
                conv_8_2_features = block(conv_7_features) 
                conv_8_2_features_d_1 = self.deconv_2(conv_8_2_features) #conv_8_2 deconvolution 1
                conv_8_2_features_d_1 = self.b_norm_2(conv_8_2_features_d_1)
                
                conv_8_2_features_d_2 = self.deconv_1(conv_8_2_features_d_1) #conv_8_2 deconvolution 2
                conv_8_2_features_d_2 = self.b_norm_1(conv_8_2_features_d_2)

                bloque_1 = torch.cat([bloque_1,conv_8_2_features_d_2],dim = 1)
                bloque_2 = torch.cat([bloque_2, conv_8_2_features_d_1] , dim = 1)
        
                bloque_3 = torch.cat([conv_8_2_features, bloque_2_pooled ], dim=1)
                bloque_3_pooled = F.max_pool2d(bloque_3, kernel_size=2, stride=2)


            if idx == 2:
                
                conv_9_2_features = block(conv_8_2_features)
                
                conv_9_2_features_d_1 = self.deconv_3(conv_9_2_features) #conv_9_2 deconvolution 1
                conv_9_2_features_d_1 = self.b_norm_3(conv_9_2_features_d_1)
                
                conv_9_2_features_d_2 = self.deconv_2(conv_9_2_features_d_1) #conv_9_2 deconvolution 2
                conv_9_2_features_d_2 = self.b_norm_2(conv_9_2_features_d_2)
                
                conv_9_2_features_d_3 = self.deconv_1(conv_9_2_features_d_2) #conv_9_2 deconvolution 3
                conv_9_2_features_d_3 = self.b_norm_1(conv_9_2_features_d_3)

                bloque_1 = torch.cat([bloque_1,conv_9_2_features_d_3],dim = 1)
                bloque_2 = torch.cat([bloque_2, conv_9_2_features_d_2] , dim = 1)
                bloque_3 = torch.cat([bloque_3, conv_9_2_features_d_1] , dim = 1)
                


                bloque_4 = torch.cat([conv_9_2_features, bloque_3_pooled ], dim=1)

            if idx == 3:
                
                output.append(bloque_1)
                output.append(bloque_2)
                output.append(bloque_3)
                output.append(bloque_4)
                
                x = block(conv_9_2_features)
                output.append(x)
                
            if idx >=4 :
                x = block(x)
                output.append(x)
                
            

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
 




def _vgg_extractor(backbone_name: str, highres: bool, progress: bool, pretrained: bool, trainable_layers: int):
    if backbone_name in backbone_urls:
        # Use custom backbones more appropriate for SSD
        arch = backbone_name.split('_')[0]
        backbone = vgg.__dict__[arch](pretrained=False, progress=progress).features
        if pretrained:
            state_dict = load_state_dict_from_url(backbone_urls[backbone_name], progress=progress)
            backbone.load_state_dict(state_dict)
    else:
        # Use standard backbones from TorchVision
        backbone = vgg.__dict__[backbone_name](pretrained=pretrained, progress=progress).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorVGG(backbone, highres)


def _DenseNet_extractor(backbone_name: str, highres: bool, progress: bool, pretrained: bool, trainable_layers: int):
    if backbone_name in backbone_urls:
        # Use custom backbones more appropriate for SSD
        arch = backbone_name.split('_')[0]
        backbone = vgg.__dict__[arch](pretrained=False, progress=progress).features
        if pretrained:
            state_dict = load_state_dict_from_url(backbone_urls[backbone_name], progress=progress)
            backbone.load_state_dict(state_dict)
    else:
        # Use standard backbones from TorchVision
        backbone = vgg.__dict__[backbone_name](pretrained=pretrained, progress=progress).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorDenseNet(backbone, highres)


def _vgg_extractor_Upgraded(backbone_name: str, highres: bool, progress: bool, pretrained: bool, trainable_layers: int):
    if backbone_name in backbone_urls:
        # Use custom backbones more appropriate for SSD
        arch = backbone_name.split('_')[0]
        backbone = vgg.__dict__[arch](pretrained=False, progress=progress).features
        if pretrained:
            state_dict = load_state_dict_from_url(backbone_urls[backbone_name], progress=progress)
            backbone.load_state_dict(state_dict)
    else:
        # Use standard backbones from TorchVision
        backbone = vgg.__dict__[backbone_name](pretrained=pretrained, progress=progress).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractorVGG_Upgraded(backbone, highres)



def ssd300_vgg16(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    """Constructs an SSD model with input size 300x300 and a VGG16 backbone.

    Reference: `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        >>> model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg_extractor("vgg16_features", False, progress, pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                           steps=[8, 16, 32, 64, 100, 300])

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    
    if pretrained:
        weights_name = 'ssd300_vgg16_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    
    return model



def ssd512_vgg16(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    """Constructs an SSD model with input size 512x512 and a VGG16 backbone."""
    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        
        pretrained_backbone = False

    
    backbone = _vgg_extractor("vgg16_features", True, progress, pretrained_backbone, trainable_backbone_layers)

    
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],  
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.2],  
          
    )

    
    defaults = {
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0], 
    }
    kwargs = {**defaults, **kwargs}

    
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)

    

    return model





def ssd300_vgg16_upgraded(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
   
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg_extractor_Upgraded("vgg16_features", False, progress, pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    
    return model

from torchvision.models import densenet121

def ssd300_DenseNet(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        
        pretrained_backbone = False

    
    densenet = densenet121(pretrained=pretrained_backbone)
    backbone = SSDFeatureExtractorDenseNet(densenet.features, highres=False)  

    
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300]
    )

    
    defaults = {
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  
    }
    kwargs = {**defaults, **kwargs}

    
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)

    return model



def ssd512_DenseNet(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        pretrained_backbone = False

    
    densenet = densenet121(pretrained=pretrained_backbone)
    backbone = SSDFeatureExtractorDenseNet(densenet.features, highres=True)  

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.2],
    )

    
    defaults = {
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  
    }
    kwargs = {**defaults, **kwargs}

    
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)

    return model

        

#Resnet

class SSDFeatureExtractorResNet(nn.Module):
    def __init__(self, backbone: resnet.ResNet, highres):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4, 
        )

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1

        backbone_out_channels = self.features[-1][-1].bn3.num_features
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            
        ])

        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ))
            
        _xavier_init(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])



#Resnet Upgraded

class SSDFeatureExtractorResNet_Upgraded(nn.Module):
    def __init__(self, backbone: resnet.ResNet):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ) 

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules(): 
            if hasattr(m, 'stride'):
                m.stride = 1

        backbone_out_channels = self.features[-1][-1].bn3.num_features 
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ), 
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ), 
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),  
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),  
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )   
        ])
        _xavier_init(extra)
        self.extra = extra


        self.deconv_1 = nn.ConvTranspose2d(in_channels= 512 , out_channels= 2048, kernel_size=2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2) 
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)
        self.b_norm_1 = nn.BatchNorm2d(2048)
        self.b_norm_2 = nn.BatchNorm2d(512)
        self.b_norm_3 = nn.BatchNorm2d(512)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        x_pooled = F.max_pool2d(x, kernel_size=2, stride=2)
        output = []
        for idx, block in enumerate(self.extra):
            if idx == 0:
                
                conv_7_features = block(x) 
                conv_7_features_d = self.deconv_1(conv_7_features)
                conv_7_features_d = self.b_norm_1(conv_7_features_d)
                bloque_1 = torch.cat([x, conv_7_features_d], dim=1)
                
                bloque_2 = torch.cat([x_pooled, conv_7_features], dim=1)
                bloque_2_pooled = F.max_pool2d(bloque_2, kernel_size=2, stride=2)
                
                
            if idx == 1:
                
                conv_8_2_features = block(conv_7_features)
                conv_8_2_features_d_1 = self.deconv_2(conv_8_2_features)
                conv_8_2_features_d_1 = self.b_norm_2(conv_8_2_features_d_1)
                
                conv_8_2_features_d_2 = self.deconv_1(conv_8_2_features_d_1)
                conv_8_2_features_d_2 = self.b_norm_1(conv_8_2_features_d_2)

                bloque_1 = torch.cat([bloque_1,conv_8_2_features_d_2],dim = 1)
                bloque_2 = torch.cat([bloque_2, conv_8_2_features_d_1] , dim = 1)
        
                bloque_3 = torch.cat([conv_8_2_features, bloque_2_pooled ], dim=1)
                bloque_3_pooled = F.max_pool2d(bloque_3, kernel_size=2, stride=2)


            if idx == 2:
                
                conv_9_2_features = block(conv_8_2_features)
                conv_9_2_features_d_1 = self.deconv_3(conv_9_2_features)
                conv_9_2_features_d_1 = self.b_norm_3(conv_9_2_features_d_1)
                
                conv_9_2_features_d_2 = self.deconv_2(conv_9_2_features_d_1)
                conv_9_2_features_d_2 = self.b_norm_2(conv_9_2_features_d_2)
                
                conv_9_2_features_d_3 = self.deconv_1(conv_9_2_features_d_2)
                conv_9_2_features_d_3 = self.b_norm_1(conv_9_2_features_d_3)

                bloque_1 = torch.cat([bloque_1,conv_9_2_features_d_3],dim = 1)
                bloque_2 = torch.cat([bloque_2, conv_9_2_features_d_2] , dim = 1)
                bloque_3 = torch.cat([bloque_3, conv_9_2_features_d_1] , dim = 1)
                


                bloque_4 = torch.cat([conv_9_2_features, bloque_3_pooled ], dim=1)

            if idx == 3:
                
                output.append(bloque_1)
                output.append(bloque_2)
                output.append(bloque_3)
                output.append(bloque_4)
                x = block(conv_9_2_features)
                output.append(x)
                
            if idx >=4 :
                x = block(x)
                output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])

class SSDFeatureExtractorResNet_Upgraded_version_2(nn.Module):
    def __init__(self, backbone: resnet.ResNet):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ) 

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules(): 
            if hasattr(m, 'stride'):
                m.stride = 1

        backbone_out_channels = self.features[-1][-1].bn3.num_features 
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ), 
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ), 
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),  
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),  
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(128, 256, kernel_size=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )   
        ])
        _xavier_init(extra)
        self.extra = extra


        self.deconv_1 = nn.ConvTranspose2d(in_channels= 512 , out_channels= 2048, kernel_size=2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2) 
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels = 2048, out_channels = 512 , kernel_size = 1)
        self.conv_2 = nn.Conv2d(in_channels = 512, out_channels = 256 , kernel_size = 1)
        self.b_norm_1 = nn.BatchNorm2d(2048)
        self.b_norm_2 = nn.BatchNorm2d(512)
        self.b_norm_3 = nn.BatchNorm2d(512)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        x_pooled = F.max_pool2d(x, kernel_size=2, stride=2)
        x_pooled = self.conv_1(x_pooled) #2048 --> 512 channels
        output = []
        for idx, block in enumerate(self.extra):
            if idx == 0:
                
                conv_7_features = block(x) 
                conv_7_features_d = self.deconv_1(conv_7_features)
                conv_7_features_d = self.b_norm_1(conv_7_features_d)
                bloque_1 = x + conv_7_features_d

                bloque_2 = x_pooled + conv_7_features
                bloque_2_pooled = F.max_pool2d(bloque_2, kernel_size=2, stride=2)
                
                
            if idx == 1:
                
                conv_8_2_features = block(conv_7_features)
                conv_8_2_features_d_1 = self.deconv_2(conv_8_2_features)
                conv_8_2_features_d_1 = self.b_norm_2(conv_8_2_features_d_1)
                
                conv_8_2_features_d_2 = self.deconv_1(conv_8_2_features_d_1)
                conv_8_2_features_d_2 = self.b_norm_1(conv_8_2_features_d_2)

                bloque_1 = bloque_1 + conv_8_2_features_d_2
                bloque_2 =  bloque_2 + conv_8_2_features_d_1
        
                bloque_3 = conv_8_2_features + bloque_2_pooled
                bloque_3_pooled = F.max_pool2d(bloque_3, kernel_size=2, stride=2)
                bloque_3_pooled = self.conv_2(bloque_3_pooled) #512 --> 256 channels


            if idx == 2:
                
                conv_9_2_features = block(conv_8_2_features)
                conv_9_2_features_d_1 = self.deconv_3(conv_9_2_features)
                conv_9_2_features_d_1 = self.b_norm_3(conv_9_2_features_d_1)
                
                conv_9_2_features_d_2 = self.deconv_2(conv_9_2_features_d_1)
                conv_9_2_features_d_2 = self.b_norm_2(conv_9_2_features_d_2)
                
                conv_9_2_features_d_3 = self.deconv_1(conv_9_2_features_d_2)
                conv_9_2_features_d_3 = self.b_norm_1(conv_9_2_features_d_3)

                bloque_1 = bloque_1 + conv_9_2_features_d_3
                bloque_2 = bloque_2 + conv_9_2_features_d_2
                bloque_3 = bloque_3 + conv_9_2_features_d_1
                


                bloque_4 = conv_9_2_features + bloque_3_pooled 

            if idx == 3:
                
                output.append(bloque_1)
                output.append(bloque_2)
                output.append(bloque_3)
                output.append(bloque_4)
                x = block(conv_9_2_features)
                output.append(x)
                
            if idx >=4 :
                x = block(x)
                output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
    
    


def _resnet_extractor(backbone_name: str, pretrained: bool, trainable_layers: int, highres: bool):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet(backbone,highres)



def _resnet_extractor_Upgraded(backbone_name: str, pretrained: bool, trainable_layers: int, load:bool ):
    
    if load: #Caso Backbone pre entrenada en SipakMed
        backbone = resnet.__dict__[backbone_name](pretrained=False)
    
        checkpoint = torch.load('/content/drive/MyDrive/Academico/TD8 personal ' + '/pesos_modelos/Resnet50_SipakMed') #Cargo los pesos de Resnet 50 pre entrenado en SipakMed
        
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('fc.weight', None)  
        state_dict.pop('fc.bias', None)    

        # Cargar el resto de los pesos
        backbone.load_state_dict(state_dict, strict=False)

        
        num_classes = 91  
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
    if not load:
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet_Upgraded(backbone)


def _resnet_extractor_Upgraded_version2(backbone_name: str, pretrained: bool, trainable_layers: int, load:bool ):
    

    if load:
        backbone = resnet.__dict__[backbone_name](pretrained=False)
    
        checkpoint = torch.load('/content/drive/MyDrive/Academico/TD8 personal ' + '/pesos_modelos/Resnet50_SipakMed') #Cargo los pesos de Resnet 50 pre entrenado en SipakMed
        
        
        state_dict = checkpoint['model_state_dict']
        state_dict.pop('fc.weight', None)  
        state_dict.pop('fc.bias', None)    

        
        backbone.load_state_dict(state_dict, strict=False)

        
        num_classes = 91  
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        print('Se cargaron los pesos de SipakMed en Resnet 50')
        
    if not load:
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet_Upgraded_version_2(backbone)





def ssd512_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        pretrained_backbone = False 
    
    
    backbone = _resnet_extractor("resnet50", pretrained_backbone, trainable_backbone_layers,True)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])


    
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)
    
    if pretrained:
        weights_name = 'ssd512_resnet50_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    
    return model


def ssd300_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        pretrained_backbone = False 

    
    
    
    backbone = _resnet_extractor("resnet50", pretrained_backbone, trainable_backbone_layers,False)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2],], #Saque el ultimo aspect ratio ya que se saca una convolucion para este caso de 300
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    
    if pretrained:
        weights_name = 'ssd300_resnet50_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    
    return model












def ssd512_resnet50_Upgraded(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, load : Optional[bool] = False ,**kwargs: Any):

    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        pretrained_backbone = False 
    
    

    backbone = _resnet_extractor_Upgraded("resnet50", pretrained_backbone, trainable_backbone_layers,load)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)

    

    
    if pretrained: #Caso en el que uso pesos de COCO que sean compatibles con la arquitectura Upgrade (no incluido en Proyecto final)


        state_dict_nuevo = model.state_dict()
        weights_name = 'ssd512_resnet50_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        

        for name, param in state_dict.items():
            if name in state_dict_nuevo and state_dict_nuevo[name].shape != param.shape: 
                
                if param.shape == torch.Size([364, 2048, 3, 3]): 
                    state_dict_nuevo[name][:, :2048, :, :] = param
                    
                if param.shape == torch.Size([16, 2048, 3, 3]): 
                    state_dict_nuevo[name][:, :2048, :, :] = param
                    
                
                
                if param.shape == torch.Size([546, 512, 3, 3]): 
                    if name == 'head.classification_head.module_list.1.weight': 
                        state_dict_nuevo[name][:, 2048:2560, :, :] = param
                    if name == 'head.classification_head.module_list.2.weight': 
                        state_dict_nuevo[name][:, 2560:3072, :, :] = param
                        
                if param.shape == torch.Size([24, 512, 3, 3]): 
                    if name == 'head.regression_head.module_list.1.weight': 
                        state_dict_nuevo[name][:, 2048:2560, :, :] = param
                    if name == 'head.regression_head.module_list.2.weight': 
                        state_dict_nuevo[name][:, 2560:3072, :, :] = param
                
                
                if param.shape == torch.Size([546, 256, 3, 3]): 
                    state_dict_nuevo[name][:, 3072: , :, :] = param

                if param.shape == torch.Size([24, 256, 3, 3]): 
                    state_dict_nuevo[name][:, 3072: , :, :] = param

            if name in state_dict_nuevo and state_dict_nuevo[name].shape == param.shape:
                state_dict_nuevo[name].copy_(param)              
                
                    
        model.load_state_dict(state_dict_nuevo)
    
    return model


def ssd512_resnet50_Upgraded_version2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, load : Optional[bool] = False ,**kwargs: Any):

    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        pretrained_backbone = False 
    
    
    backbone = _resnet_extractor_Upgraded_version2("resnet50", pretrained_backbone, trainable_backbone_layers,load)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)
    
    if pretrained:
        weights_name = 'ssd512_resnet50_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    
    return model
    






import torchvision.models as models
import torch.nn as nn

def _swin_extractor(backbone_name: str, highres: bool, pretrained: bool, trainable_layers: int):
    """
    Adaptación del extractor para usar Swin Transformer en lugar de VGG.
    """
   
    if backbone_name == "swin_t":
        backbone = models.swin_t(pretrained=pretrained)
    elif backbone_name == "swin_s":
        backbone = models.swin_s(pretrained=pretrained)
    elif backbone_name == "swin_b":
        backbone = models.swin_b(pretrained=pretrained)
    else:
        raise ValueError(f"Backbone {backbone_name} no es soportado")

    
    total_layers = len(backbone.features)
    assert 0 <= trainable_layers <= total_layers, f"trainable_layers debe estar entre 0 y {total_layers}"

    # Congelar capas que no queremos entrenar
    if trainable_layers < total_layers:
        layers_to_freeze = total_layers - trainable_layers
        for i, layer in enumerate(backbone.features[:layers_to_freeze]):
            for param in layer.parameters():
                param.requires_grad = False
                
    backbone = backbone.features  
    return SSDFeatureExtractorSwin(backbone, highres)




def ssd300_swin(name : str,pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
   
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False

    backbone = _swin_extractor(name, False,pretrained_backbone, trainable_backbone_layers)

    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                           steps=[8, 16, 32, 64, 100, 300])

    defaults = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],  
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    
    
    return model







def _convnext_extractor(backbone_name: str, highres: bool, pretrained: bool, trainable_layers: int):
    
    if backbone_name == "convnext_tiny":
        backbone = models.convnext_tiny(pretrained=pretrained)
    elif backbone_name == "convnext_small":
        backbone = models.convnext_small(pretrained=pretrained)
    elif backbone_name == "convnext_base":
        backbone = models.convnext_base(pretrained=pretrained)
    elif backbone_name == "convnext_large":
        backbone = models.convnext_large(pretrained=pretrained)
    else:
        raise ValueError(f"Backbone {backbone_name} no es soportado")

    
    total_layers = len(backbone.features)
    assert 0 <= trainable_layers <= total_layers, f"trainable_layers debe estar entre 0 y {total_layers}"

    if trainable_layers < total_layers:
        layers_to_freeze = total_layers - trainable_layers
        for i, layer in enumerate(backbone.features[:layers_to_freeze]):
            for param in layer.parameters():
                param.requires_grad = False

    backbone = backbone.features
    
    return SSDFeatureExtractorConvNeXt(backbone, highres)

def ssd300_convnext(name : str,pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    
    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False
        
    backbone = _convnext_extractor(name, False,pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2],[2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.2],
                                           steps=[4, 8, 16, 32, 64, 128, 300])

    defaults = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }
    kwargs = {**defaults, **kwargs}

    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    
    return model



def ssd512_convnext(name : str,pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                 pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    
    
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False
 
    backbone = _convnext_extractor(name, True,pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2],[2], [2]], scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.2, 1.35],)
                                           

    defaults = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],  # Las medias y desviaciones estándar de ImageNet para Swin
    }
    kwargs = {**defaults, **kwargs}

    
    model = SSD(backbone, anchor_generator, (512, 512), num_classes, **kwargs)

    
    
    return model

