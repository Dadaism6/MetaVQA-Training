#CONFIG=lavis/projects/blip2/train/advqa_t5_elm.yaml
#
#python -m torch.distributed.run \
#    --nproc_per_node=8 \
#    --master_port=10041 \
#    scripts/train.py --cfg-path $CONFIG

CONFIG=lavis/projects/blip2/train/metavqa_multiview_multiframe_test.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=10041 \
    scripts/train.py --cfg-path $CONFIG