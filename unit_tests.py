# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:52:07 2024

@author: magfrump
"""

import unittest
from scoring.mf_base_scorer import MFBaseScorer
import toy_model
import pandas as pd
import numpy as np
import mock_data_generation as mdg
    
#def rate_note_with_error(is_lie, is_twitcher, error_rate=.05):
    

class TestVariousMethods(unittest.TestCase):
    def test_post_creation(self):
        posts = mdg.create_posts(1000)
        self.assertEqual(posts.shape[0], 1000)

    def test_contrib_creation(self):
        contribs = mdg.create_contributors(1000,.02)
        self.assertEqual(contribs.shape[0],1000)
        
    def test_create_notes_dataset(self):
        # Setup
        contributors_data = pd.DataFrame({
            'contributor_id': [1, 2],
            'type': ['birder', 'twitcher']
        })
        posts_data = pd.DataFrame({
            'topic': ['Politics', 'Environment'],
            'blatant_lie': [0, 1]
        })
        # Execute
        result = mdg.create_notes_dataset(contributors_data, posts_data, 0.1, 10, 1)
        # Test
        self.assertFalse(result.empty)
        self.assertTrue('contributor_id' in result.columns)
        self.assertTrue('flag' in result.columns)
        self.assertTrue(result['flag'].isin([0, 1]).all())

    def test_create_ratings_dataset(self):
        # Setup
        contributors_data = mdg.create_contributors(100, .1)
        posts_data = mdg.create_posts(100)
        notes_data = mdg.create_notes_dataset(contributors_data, posts_data)
    
        # Execute
        ratings = mdg.create_ratings_dataset(contributors_data, notes_data, attention_span=10, multiplier=1)
    
        # Test
        self.assertEqual(len(ratings),1000)
                
    def test_mf_on_data(self):
        toy = toy_model.ToyModel()
        contributors_data = mdg.create_contributors(100, .1)
        posts_data = mdg.create_posts(100)
        notes_data = mdg.create_notes_dataset(contributors_data, posts_data)
    
        # Execute
        ratings = mdg.create_ratings_dataset(contributors_data, notes_data, attention_span=10, multiplier=1)
        ratings["raterParticipantId"] = ratings["rater_id"]
        ratings["helpfulNum"] = ratings["rate_helpful"]
        toy.load_dataframe(ratings)
        self.assertTrue(toy.run_mf())
        toy.report()
        
    def test_df_loading(self):
        toy = toy_model.ToyModel()
        df = pd.DataFrame()
        df["Numbers"] = pd.Series([1,2,3])
        toy.load_dataframe(df)
        self.assertEqual(
            pd.testing.assert_frame_equal(df, toy.get_dataframe()), None)

    def test_matrix_factorization(self):
        toy = toy_model.ToyModel()
        df = pd.DataFrame(
            {"noteId": [1,1,1,1,2,2,2,2,3,3,3,3],
             "raterParticipantId": [1,2,3,4,1,2,3,4,1,2,3,4],
             "helpfulNum": [1,1,0,0,.5,1,1,.5,0,0,1,1]}
            )
        toy.load_dataframe(df)
        self.assertTrue(toy.run_mf())
        print("fnord 1")
        toy.report()

    def test_matrix_factorization(self):
        toy = toy_model.ToyModel()
        toy.load_dataframe(pd.read_csv("mock_data.csv"))
        self.assertTrue(toy.run_mf())
        print("fnord 2")
        values = toy.report()
        print(values[0]['internalNoteIntercept'])
        
    def test_one_dim_moderate_statement_wins(self):
        toy = toy_model.ToyModel(1)
        toy.load_dataframe(pd.read_csv("mock_data_0.csv"))
        self.assertTrue(toy.run_mf())
        print("fnord 3")
        note_scores = toy.report()[0]['internalNoteIntercept']
        self.assertEqual(note_scores[1],note_scores.max())
    
    

if __name__ == '__main__':
    unittest.main()
    
# central test case at base level for exploration
toy = toy_model.ToyModel(1)
contributors_data = mdg.create_contributors(1000, .1)
posts_data = mdg.create_posts(1000)
notes_data = mdg.create_notes_dataset(contributors_data, posts_data)

# Execute
ratings = mdg.create_ratings_dataset(contributors_data, notes_data, attention_span=10, multiplier=1)
ratings["raterParticipantId"] = ratings["rater_id"]
ratings["helpfulNum"] = ratings["rate_helpful"]
toy.load_dataframe(ratings)
toy.run_mf()
report = toy.report()

# Merge "ground truth" values such as "blatant lies" and "twitcher/birder" with model output
merged_notes_data = report[0].merge(notes_data, on="noteId", how="inner", sort=True)
contributors_data["raterParticipantId"] = contributors_data["contributor_id"]
merged_user_data = report[1].merge(contributors_data, on="raterParticipantId", how="inner", sort=True)