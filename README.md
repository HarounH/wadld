# wadld
DOOM WAD Level Designer, aka wadld (using ml)

# Requirements
- Python 3.7 (Anaconda), PyTorch 1.0, no backward compatibility guaranteed.
- html5lib for parsing websites to get data (`pip install html5lib` to install it)
- Install pandas, requests, and bs4.
- omgifol for some WAD handling
- OpenCV2 (`pip install opencv-python`)

## Data
We get user generated WAD files from https://www.doomworld.com/idgames/levels/doom/.
`cd data`, `python download_user_wads.py`

### Processing data: Binarize
From base dir: `python -m data.preprocess data/dataset/all_wads.pkl -o data/preprocessed_data/binarized.pkl -conf data/default_preprocess_conf.json`
<!--- The steps inside the comment are no longer necessary.
followed by `sh manual_fix.sh` (hit `y` or `A` whenever prompted) followed by `python download_user_wads.py --reconstruct`.
-->

## WAD Parser
We could use https://github.com/devinacker/omgifol
