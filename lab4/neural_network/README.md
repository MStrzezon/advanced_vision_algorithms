# Thermo detector based on deep convolutional neural network
We can use three methods of data fusion:
1. Early fusion - we concatenate RGB and thermal images and then we fed a detector
2. Middle fusion - the detector initially processes images separately and at a certain stage of the neural network features
representing both images are combined
3. Late fusion - at the output we get separate results for each of the detectors that are combined by finding matching
surrounding boxes and averaging their parameters
## Run neural network
1. Set in [thermo_detection_pl.py](thermo_detection_pl.py) in section `METHOD` variable `FUSION` to `EARLY` or `LATE`
2. Run script
```bash
python thermo_detection_pl.py
```