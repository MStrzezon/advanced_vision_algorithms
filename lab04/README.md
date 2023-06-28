# Thermovision

## Solution files

| Exercise number | File name / Folder name                                | Description                                                                            |
|-----------------|--------------------------------------------------------|----------------------------------------------------------------------------------------|
| 1               | [silhouette_detector.py](silhouette_detector.py)       | Simple detection of objects using thresholding and object analysis                     |
| 2               | [neural_network](neural_network)                       | The use of a deep convolutional neural network and RGB and thermal imaging data fusion |
| 3               | [probabilistic_detector.py](probabilistic_detector.py) | Application of a probabilistic pattern                                                 |


## Requirements

1. `vid1_IR.avi` - thermovision video:

![frame_003090.png](resources%2Fframes%2Fframe_003090.png)
## Probabilistic detector guide
You need to run 3 steps:
1. Save samples and example frame
```bash
python3 probabilistic_detector.py --save
```
2. Create pattern
```bash
python3 probabilistic_detector.py --create
```
3. Detect objects
```bash
python3 probabilistic_detector.py --detect
```
