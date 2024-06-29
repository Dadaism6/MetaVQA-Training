#CONFIG=lavis/projects/blip2/train/advqa_t5_elm.yaml
#
#python -m torch.distributed.run \
#    --nproc_per_node=8 \
#    --master_port=10041 \
#    scripts/train.py --cfg-path $CONFIG

CONFIG=lavis/projects/blip2/eval/eval_metavqa_multiview_mixmultiframe_quarter.yaml

CUDA_VISIBLE_DEVICES=6,7  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=10052 \
    scripts/train.py --cfg-path $CONFIG