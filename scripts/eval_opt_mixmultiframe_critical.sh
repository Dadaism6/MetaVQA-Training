#CONFIG=lavis/projects/blip2/train/advqa_t5_elm.yaml
#
#python -m torch.distributed.run \
#    --nproc_per_node=8 \
#    --master_port=10041 \
#    scripts/train.py --cfg-path $CONFIG

CONFIG=lavis/projects/blip2/eval/eval_metavqa_multiview_opt_mixmultiframe_critical_test.yaml

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=10050 \
    scripts/train.py --cfg-path $CONFIG