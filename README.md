# Detection Anomal

### Prepare features at [***I3D_extractor***](./I3D_extractor/)

### Infer video at [***RTFM***](./RTFM)
To infer video, firstly you need to save this checkpoint `/media/v100/DATA4/thviet/AnomalyDetection/RTFM/ckpt/rtfm-NAdam-i3d_v2.pkl` to folder `ckpt`.
After getting `.npy` file of embedded video at `../I3D_extractor/output`, run this to get the score-per-frames.
```shell
cd RTFM
python3 infer.py --video_feats 'npy-path' --scores_file 'score-per-frames'
```

### Process videos
To put scores to videos, run this
```shell
cd ..
python3 process_videos.py
```
Note that the input videos are saved at `/I3D_extractor/demovideos` while their scores are at `/RTFM/scores` and the final videos are at `./out_videos`.