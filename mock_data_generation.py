#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:31:19 2024

@author: magfrump
"""
from scoring.mf_base_scorer import MFBaseScorer
import toy_model
import pandas as pd
import numpy as np


def create_posts(n_posts, seed = 1224):
    np.random.seed(seed)
    post_ids = range(n_posts)
    topics = np.random.choice(
        a=["Formula One","Coffee","Data Science","Gardening","Politics"],
        size=n_posts,
        p=[.2, .05, .3, .2, .25])
    lies = np.random.choice(
        a=[0,1],
        size = n_posts,
        p=[.9,.1])
    posts = pd.DataFrame({"post_id":post_ids,"topic":topics,"blatant_lie":lies})
    return posts

def create_contributors(n_contribs, twitcher_probability, seed = 1224):
    np.random.seed(seed)
    contrib_ids = range(n_contribs)
    contrib_types = np.random.choice(
        a=["birder","twitcher"],
        size=n_contribs,
        p=[1-twitcher_probability, twitcher_probability])
    contribs = pd.DataFrame({"contributor_id":contrib_ids, "type":contrib_types})
    return contribs

def create_notes_dataset(contributors_data_frame, posts_data, param_gamma=0.1, attention_span=10, multiplier=1, seed=1224):
    np.random.seed(seed)
    # Filter data frames by type
    birder_df = contributors_data_frame[contributors_data_frame['type'] == 'birder']
    twitcher_df = contributors_data_frame[contributors_data_frame['type'] == 'twitcher']

    # Calculate number of birders and twitchers
    n_birders = len(birder_df)
    n_twitchers = len(twitcher_df)

    # Sample posts for birders
    birder_notes = posts_data.sample(n=attention_span * n_birders, replace=True)
    birder_notes['contributor_id'] = np.repeat(birder_df['contributor_id'].values, attention_span)
    birder_notes['error'] = np.random.binomial(1, 0.05, size=len(birder_notes))
    birder_notes['flag'] = np.where(birder_notes['error'] == 0, birder_notes['blatant_lie'].astype(int), (1 - birder_notes['blatant_lie']).astype(int))
    birder_notes['whistle'] = 0
    birder_notes['noteId'] = range(n_birders*attention_span)
    birder_notes = birder_notes[birder_notes['flag'] == 1]

    # Sample posts for twitchers on non-target topics
    non_target_notes = posts_data[posts_data['topic'] != 'Politics'].sample(n=int(round(multiplier * attention_span * (1 - param_gamma)) * n_twitchers), replace=True)
    non_target_notes['contributor_id'] = np.repeat(twitcher_df['contributor_id'].values, int(round(multiplier * attention_span * (1 - param_gamma))))
    non_target_notes['error'] = np.random.binomial(1, 0.05, size=len(non_target_notes))
    non_target_notes['flag'] = np.where(non_target_notes['error'] == 0, non_target_notes['blatant_lie'].astype(int), (1 - non_target_notes['blatant_lie']).astype(int))
    non_target_notes['whistle'] = 0
    non_target_notes['noteId'] = range(n_birders*attention_span, n_birders*attention_span+non_target_notes.shape[0])
    non_target_notes = non_target_notes[non_target_notes['flag'] == 1]

    # Sample posts for twitchers on target topics
    target_notes = posts_data[posts_data['topic'] == 'Politics'].sample(n=int(round(multiplier * attention_span * param_gamma) * n_twitchers), replace=True)
    target_notes['contributor_id'] = np.repeat(twitcher_df['contributor_id'].values, int(round(multiplier * attention_span * param_gamma)))
    target_notes['flag'] = np.where(target_notes['blatant_lie'] == 0, 1, 0)
    target_notes['whistle'] = 1
    target_notes['noteId'] = range(n_birders*attention_span+non_target_notes.shape[0], n_birders*attention_span+non_target_notes.shape[0]+target_notes.shape[0])
    target_notes = target_notes[target_notes['flag'] == 1]

    # Combine all notes
    notes = pd.concat([birder_notes, non_target_notes, target_notes])
    return notes

def create_ratings_dataset(contributors_data_frame, notes_data, attention_span=30, multiplier=1, seed=1224):
    np.random.seed(seed)
    n_birders = contributors_data_frame[contributors_data_frame['type'] == 'birder'].shape[0]
    n_twitchers = contributors_data_frame[contributors_data_frame['type'] == 'twitcher'].shape[0]

    # Get birder ratings
    birder_ratings = notes_data.sample(n=attention_span * n_birders, replace=True)
    birder_ratings['rater_id'] = pd.Series(np.repeat(contributors_data_frame[contributors_data_frame['type'] == 'birder']['contributor_id'].values, attention_span))
    birder_ratings['error'] = np.random.binomial(1, 0.05, size=attention_span * n_birders)
    birder_ratings['rate_helpful'] = np.where(birder_ratings['error'] == 0, birder_ratings['blatant_lie'].astype(int), (1 - birder_ratings['blatant_lie']).astype(int))

    # Get twitcher ratings
    twitcher_ratings = notes_data[notes_data['whistle'] == 1].sample(n=int(multiplier * attention_span * n_twitchers), replace=True)
    twitcher_ratings['rater_id'] = pd.Series(np.repeat(contributors_data_frame[contributors_data_frame['type'] == 'twitcher']['contributor_id'].values, int(multiplier * attention_span)))
    twitcher_ratings['rate_helpful'] = 1

    # Combine ratings
    ratings = pd.concat([birder_ratings, twitcher_ratings])
    return ratings