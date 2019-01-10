# wadld
DOOM WAD Level Designer, aka wadld (using ml)

# Requirements
- Everything uses Python3, PyTorch 1.0, no backward compatibility guaranteed.
## Data
We get user generated WAD files from https://www.doomworld.com/idgames/levels/doom/ . No, we don't want you to manually download each one - we provide a script.
`python data/download_user_wads.py --help`
## WAD Parser
We use https://github.com/devinacker/omgifol
