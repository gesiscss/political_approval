This repositiory contains the code for the paper "On the Reliability and Validity of Measuring Approval of Political Actors" at EMNLP 2020

Install requirements with `pip install -r requirements.txt`

To replicate the results in the paper, run `replicate results.ipynb`

To rerun the analysis:

0. rehydrate the tweets based on the IDs in `data/all_data_no_tweet_text.csv`
1. run `train_in_domain_models.py`
2. run DSSD using `./run_dssd.sh`
3. run `label_stance.py`