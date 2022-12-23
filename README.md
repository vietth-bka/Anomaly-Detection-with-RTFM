This repo is heavily inspired from [RTFM](https://github.com/tianyu0207/RTFM) and [I3D_extractor](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet). 
## Training RTFM 
Authors of `RTFM` seem to have no intentions to publish the implementation of the input processing so this makes it impossible to reproduce the result of the paper.

## Inference an input video 

### Prepare features at [I3D_extractor](./I3D_extractor/)
We added file `extract_features_v2.py` to `I3D_extractor` then proactively generate the video features used for training `RTFM`.

### Get score-per-frames with [RTFM](./RTFM)
To infer video, firstly you might need to save this [checkpoint](https://drive.google.com/file/d/1ocvSevEtlXdajpILMQp5ub9954E3AE7B/view?usp=share_link) (or you can train it by yourself) to folder `ckpt`.
After getting `.npy` file of embedded video at `../I3D_extractor/output`, run this to get the score-per-frames.
```shell
cd RTFM
python3 infer.py --video_feats 'npy/path' --save_scores './RTFM/scores/scores_file_name.npy'
```

### Process videos
To put scores to videos, run this
```shell
cd ..
python3 process_videos.py --input 'path/to/input-video' --scores './RTFM/scores/scores_file_name.npy' --out 'path/to/video-out'
```
Note that the input videos are saved at `/I3D_extractor/demovideos` while their scores are at `/RTFM/scores` and the final videos are at `./out_videos`.

### To-do list
- [x] Release the extract-feature code
- [ ] Publish the training results