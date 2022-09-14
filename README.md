# CRIF

## Description
This repository is the source code of the paper *Learning to Infer User Implicit Preference in Conversational Recommendation* implemented via PyTorch.

More descriptions are available via the paper.

The code is partially refer to *Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation* ([FPAN](https://github.com/xxkkrr/FPAN)).

## Requirement
- `torch==1.4.0`   
- `torch_geometric==1.4.3`
- `tqdm`
- `sklearn`

## Dataset
The dataset we used is based on [Yelp](https://www.yelp.com/dataset/) and [LastFM](https://grouplens.org/datasets/hetrec-2011/) which are processed in [SCPR](https://github.com/farrecall/SCPR)

## Usage
* **To train CRIF offline model:**
  ``python offline_train_rec.py``
  The model paramters will be saved in `\recommendersystem\recmodel`
* **To train policy network in conversational component:**
  Inverse reinforcement learning (IRL) is adopted to tackle the decision-making problem. We conduct our approach to learn a reward function from human feedback and optimize it explicitly using human feedback. At the same time, we also train a policy to maximize reward given the current predicted reward function. 
  - `python train_agent_ear.py --mode pretrain`: train policy network without pretrain model.
    The model parameters will be saved in `\agents\agent_ear`
  - `python train_agent_ear.py --mode PG`: further train policy network with pretrain model.
    The model parameters will be saved in `\agents\agent_ear`
* **To directly evaluate CRIF:**
  We provided the model parameters in corresponding folder.
  - `python offline_test_rec.py`: evaluate CRIF offline model
  - `python test_agent_ear.py`: evaluate CRIF with user simulator
