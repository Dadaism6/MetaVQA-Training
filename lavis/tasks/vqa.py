"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import wandb
import torch
import torch.distributed as dist
import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger, SmoothedValue
import numpy as np
import copy, random


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()
    def pass_wandb(self, wandb_run):
        self.wandb = wandb_run
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            self.wandb.log({"train_epoch": epoch,
                            "train_inner_epoch": inner_epoch,
                            "train_iters": i,
                            "train_loss": loss.detach().cpu(),
                            "train_lr": optimizer.param_groups[0]["lr"]})

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def evaluation(self, model, data_loader, cuda_enabled=True, cur_epoch=None, max_batches=None, randomize=False):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        max_batches = None
        results = []
        # text_table = wandb.Table(columns=["eval_epoch", "question", "correct_answer", "predict_answer"])

        # Initialize a list to store batch indices if randomization is enabled
        if randomize and max_batches is not None:
            total_batches = len(data_loader)  # Total number of batches in the DataLoader
            print(f"total batches is: {total_batches}")
            batch_indices = list(range(total_batches))
            random.shuffle(batch_indices)  # Shuffle the list of indices
            batch_indices = batch_indices[:max_batches]  # Take only the first max_batches indices

        processed_batches = 0  # Counter for batches actually processed (for randomization)

        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if max_batches is not None and processed_batches >= max_batches:
                break  # Stop evaluation after processing max_batches batches

            # Check if the current batch index is in the randomly selected indices
            if randomize and max_batches is not None:
                if i not in batch_indices:
                    continue  # Skip this batch if it's not in the selected indices

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples, epoch=cur_epoch)
            results.extend(eval_output)

            processed_batches += 1  # Increment batch processed counter

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

        # def evaluation(self, model, data_loader, cuda_enabled=True, cur_epoch = None):
    #     metric_logger = MetricLogger(delimiter="  ")
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = 10
    #
    #     results = []
    #     # text_table = wandb.Table(columns=["eval_epoch", "question", "correct_anwser", "predict_anwser"])
    #     for samples in metric_logger.log_every(data_loader, print_freq, header):
    #         samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #
    #         eval_output = self.valid_step(model=model, samples=samples, epoch=cur_epoch)
    #         # for each in eval_output:
    #         #     text_table.add_data(each["eval_epoch"], each["question"], each["correct_anwser"], each["predict_anwser"])
    #         results.extend(eval_output)
    #     # self.wandb.log({"eval_result": text_table})
    #     if is_dist_avail_and_initialized():
    #         dist.barrier()

        return results
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if hasattr(dataset[split], "coco_fmt_qust_file") and dataset[split].coco_fmt_qust_file is not None:
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(self.anno_files), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples, epoch):

        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        full_pairs = []
        question_id = samples["questions"]
        correct_anwsers = samples["answers"]
        data_source = ["" for _ in range(len(question_id))]
        question_type = ["" for _ in range(len(question_id))]
        answer_form = ["" for _ in range(len(question_id))]
        if "data_source" in samples:
            data_source = samples["data_source"]
        if "question_type" in samples:
            question_type = samples["question_type"]
        if "answer_form" in samples:
            answer_form = samples["answer_form"]
        # for answer, ques_id, correct_anwser in zip(answers, question_id, correct_anwsers):
        for answer, ques_id, correct_anwser, ds, qt, af in zip(answers, question_id, correct_anwsers, data_source, question_type, answer_form):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            # pred_qa_pairs.append({"question_id": ques_id, "answer": answer})
            full_pairs.append({"eval_epoch": epoch,
                               "question": ques_id, 
                               "correct_anwser": correct_anwser, 
                               "predict_anwser": answer,
                               "data_source": ds,
                               "question_type": qt,
                               "answer_form": af
                               })

        return full_pairs

    def after_evaluation(self, val_result, split_name, epoch):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_{epoch}_vqa_result",
            # remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(resFile=result_file, quesFile=self.ques_files[split])

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info("%s : %.02f" % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type]))
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics


# @registry.register_task("gqa")
# GQATask = VQATask
# @registry.register_task("aok_vqa")
# AOKVQATask = VQATask
@registry.register_task("gqa")
class GQATask(VQATask):
    pass
    # def valid_step(self, model, samples):
    #    answers = model.predict_answers(
    #        samples=samples,
    #        answer_list=self.answer_list,
    #        inference_method=self.inference_method,
    #        num_beams=self.num_beams,
    #        max_len=self.max_len,
    #        min_len=self.min_len,
    #        num_ans_candidates=self.num_ans_candidates,
    #        prompt=self.prompt,
    #    )
    #    pred_qa_pairs = []

    #    question_id = samples["question_id"]
    #    gt_answers = samples["answer"]

    #    for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
    #        ques_id = int(ques_id.item())
    #        pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

    #    return pred_qa_pairs

    # @dist_utils.main_process
    # def _report_metrics(self, result_file, split):
    #    """
    #    TODO: add other evaluation metrics for GQA
    #    """

    #   results = json.load(open(result_file, "r"))
    #   acc = []
    #   vqa_tool = VQAEval()

    #  for res in results:
    #      if res["gt_ans"] is None:
    # prepare test results for leaderboard evaluation
    #          self._save_result_leaderboard(results)
    #          return

    #     gt_ans = res["gt_ans"]
    #     pred = res["pred_ans"]

    #     if self.inference_method == "generate":
    #         pred = vqa_tool.processPunctuation(pred)
    #         pred = vqa_tool.processDigitArticle(pred)

    #    vqa_acc = 1 if pred == gt_ans else 0

    #    acc.append(vqa_acc)

    # accuracy = sum(acc) / len(acc) * 100
    # metrics = {"agg_metrics": accuracy, "acc": accuracy}

    # with open(
    #    os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    # ) as f:
    #    f.write(json.dumps(metrics) + "\n")

    # logging.info(metrics)

    # return metrics


@registry.register_task("3d_vqa")
class ThreeDVQATask(VQATask):
    pass


#    def valid_step(self, model, samples):
#        answers = model.predict_answers(
#            samples=samples,
#            answer_list=self.answer_list,
#            inference_method=self.inference_method,
#            num_beams=self.num_beams,
#            max_len=self.max_len,
#            min_len=self.min_len,
#            num_ans_candidates=self.num_ans_candidates,
#        )

#        pred_qa_pairs = []

#        question_id = samples["question_id"]
#        gt_answers = samples["direct_answers"]

#        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
#            pred_qa_pairs.append(
#                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
#            )

#        return pred_qa_pairs

#    @dist_utils.main_process
#    def _report_metrics(self, result_file, split):
#        """
#        Implementing accuracy computation for AOKVQA, see
#        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
#        """
# TODO add evaluation for multi-choice

#        results = json.load(open(result_file, "r"))
#        acc = []

#        for res in results:
#            if res["gt_ans"] is None:
# prepare test results for leaderboard evaluation
#                self._save_result_leaderboard(results)
#                return

#            pred = res["pred_ans"]
#            gt_ans = res["gt_ans"]

#            num_match = sum([pred == gt for gt in gt_ans])
#            vqa_acc = min(1.0, num_match / 3.0)

#            acc.append(vqa_acc)
#
#        accuracy = sum(acc) / len(acc) * 100
#        metrics = {"agg_metrics": accuracy, "acc": accuracy}

#        with open(
#            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
#        ) as f:
#            f.write(json.dumps(metrics) + "\n")

#        logging.info(metrics)

#        return metrics

#    @dist_utils.main_process
#    def _save_result_leaderboard(self, results):
#        """
#        Saving the results in the format required for leaderboard evaluation.

#        [TODO] add support for multi-choice.
#        """
#        result_leaderboard = dict()
#        for res in results:
#            result_leaderboard[res["question_id"]] = {
#                "direct_answer": res["pred_ans"],
#                "multiple_choice": "",
#            }

#        result_file = registry.get_path("result_dir") + "_leaderboard.json"

#        with open(result_file, "w") as f:
#            json.dump(result_leaderboard, f)

#        logging.info(f"Saved results for leaderboard evaluation at {result_file}")
