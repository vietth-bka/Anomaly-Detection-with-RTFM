This repo is heavily inspired from [RTFM](https://github.com/tianyu0207/RTFM) and [I3D_extractor](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet).
## Training RTFM 
Authors of `RTFM` seem to have no intentions to publish the implementation of the input processing so this makes it impossible to reproduce the result of the paper. This repository includes all you need to train and infer anomaly detection with `RTFM`.

## Inference with RTFM

### Prepare features at [I3D_extractor](./I3D_extractor/)
We added file `extract_features_v2.py` to `I3D_extractor` then proactively generate the video features used for training `RTFM` with [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/).

### Get scores-per-frame with [RTFM](./RTFM) and process the video
To infer video, firstly you might need to save this [checkpoint](https://drive.google.com/file/d/1ocvSevEtlXdajpILMQp5ub9954E3AE7B/view?usp=share_link) (or you can train it by yourself) to folder `ckpt`.
After getting `.npy` file of embedded video saved at `../I3D_extractor/output`, run this to get the scores-per-frame.
```shell
cd RTFM
python3 infer.py --video_feats 'path/to/npy' --input_video 'path/to/input-video' --output_video 'path/to/output-video'
```
#### Parameters
<pre>
--video_feats:      path of the extracted video features
--input_video:      path of the input video
--output_video:     path of the output video
</pre>


### To-do list
- [x] Release the extract-feature code.
- [ ] Publish the training results with UCF-Crime.