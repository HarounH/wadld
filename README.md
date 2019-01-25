# wadld
DOOM WAD Level Designer, aka wadld (using ml)

# Requirements
- Everything uses Python3 (Anaconda), PyTorch 1.0, no backward compatibility guaranteed.
- html5lib for parsing websites to get data (`pip install html5lib` to install it)

## Data
We get user generated WAD files from https://www.doomworld.com/idgames/levels/doom/.
`cd data`, `python download_user_wads.py`
<!--- The steps inside the comment are no longer necessary.
followed by `sh manual_fix.sh` (hit `y` or `A` whenever prompted) followed by `python download_user_wads.py --reconstruct`.
-->

## WAD Parser
We could use https://github.com/devinacker/omgifol
