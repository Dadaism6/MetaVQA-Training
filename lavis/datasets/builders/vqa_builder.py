"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.elm_datasets import ELMDataset, ELMDatasetEvalDataset
from lavis.datasets.datasets.metavqa_pretrain_dataset import MetaVQADataset,MetaVQAEvalDataset
from lavis.datasets.datasets.metavqa_multiview import MetaVQAMultiviewDataset, MetaVQAMultiviewEvalDataset
from lavis.datasets.datasets.metavqa_multiview_debug import MetaVQAMultiviewDatasetDebug, MetaVQAMultiviewEvalDatasetDebug
from lavis.datasets.datasets.metavqa_multiview_multiframe import MetaVQAMultiviewMultiFrameDataset, MetaVQAMultiviewMultiFrameEvalDataset
from lavis.datasets.datasets.metavqa_multiview_mixmultiframe import MetaVQAMultiviewMixMultiFrameDataset, MetaVQAMultiviewMixMultiFrameEvalDataset
from lavis.datasets.datasets.metavqa_multiview_mixmultiframe_critical import MetaVQAMultiviewMixMultiFrameCriticalDataset, MetaVQAMultiviewMixMultiFrameCriticalEvalDataset
from lavis.datasets.datasets.eval_metavqa_multiview_mixmultiframe import EvalMetaVQAMultiviewMixMultiFrameDataset, EvalMetaVQAMultiviewMixMultiFrameEvalDataset
from lavis.datasets.datasets.eval_metavqa_multiview_mixmultiframe_critical import EvalMetaVQAMultiviewMixMultiFrameCriticalDataset, EvalMetaVQAMultiviewMixMultiFrameCriticalEvalDataset
from lavis.datasets.datasets.eval_metavqa_multiview import EvalMetaVQAMultiviewDataset, EvalMetaVQAMultiviewEvalDataset
from lavis.datasets.datasets.eval_metavqa_multiview_mixmultiframe_critica_real import EvalMetavqaRealDataset, EvalMetavqaRealEvalDataset
from lavis.datasets.datasets.waymo_real_mix import MetavqaRealDataset, MetavqaRealEvalDataset
from lavis.datasets.datasets.metavqa_safety_finetune import FinetuneSafetyDataset, FinetuneSafetyEvalDataset

@registry.register_builder("safety_finetune")
class SafetyFinetuneBuilder(BaseDatasetBuilder):
    train_dataset_cls = FinetuneSafetyDataset
    eval_dataset_cls = FinetuneSafetyEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/safety_finetune/defaults.yaml"}
@registry.register_builder("meta_vqa_real")
class EvalMetaVQARealBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetavqaRealDataset
    eval_dataset_cls = MetavqaRealEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_real/defaults.yaml"}
@registry.register_builder("eval_meta_vqa_real")
class EvalMetaVQARealBuilder(BaseDatasetBuilder):
    train_dataset_cls = EvalMetavqaRealDataset
    eval_dataset_cls = EvalMetavqaRealEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/eval_metavqa_real/defaults.yaml"}

@registry.register_builder("eval_meta_vqa_multiview")
class EvalMetaVQAMultiviewBuilder(BaseDatasetBuilder):
    train_dataset_cls = EvalMetaVQAMultiviewDataset
    eval_dataset_cls = EvalMetaVQAMultiviewEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/eval_metavqa_multiview/defaults.yaml"}

@registry.register_builder("eval_meta_vqa_multiview_mixmultiframe_critical")
class EvalMetaVQAMultiviewCriticalBuilder(BaseDatasetBuilder):
    train_dataset_cls = EvalMetaVQAMultiviewMixMultiFrameCriticalDataset
    eval_dataset_cls = EvalMetaVQAMultiviewMixMultiFrameCriticalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/eval_metavqa_multiview_mixmultiframe_critical/defaults.yaml"}

@registry.register_builder("eval_meta_vqa_multiview_mixmultiframe")
class EvalMetaVQAMultiviewBuilder(BaseDatasetBuilder):
    train_dataset_cls = EvalMetaVQAMultiviewMixMultiFrameDataset
    eval_dataset_cls = EvalMetaVQAMultiviewMixMultiFrameEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/eval_metavqa_multiview_mixmultiframe/defaults.yaml"}


@registry.register_builder("meta_vqa_multiview_mixmultiframe_critical")
class MetaVQAMultiviewCriticalBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQAMultiviewMixMultiFrameCriticalDataset
    eval_dataset_cls = MetaVQAMultiviewMixMultiFrameCriticalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_multiview_mixmultiframe_critical/defaults.yaml"}

@registry.register_builder("meta_vqa_multiview_mixmultiframe")
class MetaVQAMultiviewMixBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQAMultiviewMixMultiFrameDataset
    eval_dataset_cls = MetaVQAMultiviewMixMultiFrameEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_multiview_mixmultiframe/defaults.yaml"}
@registry.register_builder("meta_vqa_multiview_multiframe")
class MetaVQAMultiviewMultiframeBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQAMultiviewMultiFrameDataset
    eval_dataset_cls = MetaVQAMultiviewMultiFrameEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_multiview_multiframe/defaults.yaml"}
@registry.register_builder("meta_vqa_multiview")
class MetaVQAMultiviewBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQAMultiviewDataset
    eval_dataset_cls = MetaVQAMultiviewEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_multiview/defaults.yaml"}

@registry.register_builder("meta_vqa_multiview_debug")
class MetaVQAMutliviewDebugBuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQAMultiviewDatasetDebug
    eval_dataset_cls = MetaVQAMultiviewEvalDatasetDebug

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa_multiview_debug/defaults.yaml"}
@registry.register_builder("meta_vqa")
class MetaVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MetaVQADataset
    eval_dataset_cls = MetaVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/metavqa/defaults.yaml"}

@registry.register_builder("elm")
class ELMBuilder(BaseDatasetBuilder):
    train_dataset_cls = ELMDataset
    eval_dataset_cls = ELMDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/elm/defaults.yaml"}

