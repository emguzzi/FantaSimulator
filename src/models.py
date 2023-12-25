import pymc as pm
import numpy as np
import pandas as pd


def beta_hierarchial_model(df: pd.DataFrame, mean: np.float64,
                           sd: np.float64) -> pm.Model:
    """Return a hierarchical model where the score is assumed to be beta-distr. 

    Args:
        df (pd.DataFrame): input df containing the scores of each team
        mean (np.float64): mean of the points scored by all the teams
        sd (np.float64): sd oof the points scored by all the teams

    Returns:
        pm.model.core.Model: pm.Model, can be used for sampling
    """
    teams = df["name"].unique()
    with pm.Model() as model:

        # define the hyperprior
        lambda_hp = pm.Uniform("lambda_hp", lower=0, upper=1)

        for team in teams:

            # extract the data for the team
            df_team = df.loc[df["name"] == team]
            score_team = df_team.values[0][2:-1].astype("float")
            score_team = score_team[~np.isnan(score_team)]
            score_team = score_team[:, None]
            # set the prior
            alpha = pm.Deterministic(f"alpha_{team}",
                                     1+pm.Exponential(f"Exp_alpha_{team}",
                                                      lambda_hp))
            beta = pm.Deterministic(f"beta_{team}",
                                    1+pm.Exponential(f"Exp_beta_{team}",
                                                     lambda_hp))

            # normalize the observed score
            score_obs_normalized = (score_team - mean)/(sd**2)+0.5
            too_small_scores = (score_obs_normalized[:, 0] < 0).sum()
            too_large_scores = (score_obs_normalized[:, 0] > 1).sum()
            score_obs_normalized = np.maximum(0, score_obs_normalized)
            score_obs_normalized = np.minimum(1, score_obs_normalized)

            # print(score_obs_normalized)

            if too_small_scores > 0:
                print(team, f": {too_small_scores} scores capped from below")
                print(score_obs_normalized)
            if too_large_scores > 0:
                print(team, f": {too_large_scores} scores capped from above")
                print(score_obs_normalized)

            # model the score
            normalized_score = pm.Beta(f"normalized_score_{team}", alpha=alpha, beta=beta,
                                       observed=score_obs_normalized)
            # reassemble the actual score for later sampling
            score = pm.Deterministic(f"score_{team}", mean + (sd**2)*(
                (normalized_score-0.5)))

    return model


def beta_hierarchial_model_cap(df: pd.DataFrame, mean: np.float64,
                               sd: np.float64, cap=0.2) -> pm.Model:
    """Return a hierarchical model where the score is assumed to be beta-distr.

    The model also allow for a capping factor. To avoid too much porbability
    at the end of the interval (very high and very low scores) a capping is 
    introduced. The normalized scores can only go from 0.2 to 0.8 (instead of 
    [0,1]). This seems to facilitate the convergence

    Args:
        df (pd.DataFrame): input df containing the scores of each team
        mean (np.float64): mean of the points scored by all the teams
        sd (np.float64): sd oof the points scored by all the teams

    Returns:
        pm.model.core.Model: pm.Model, can be used for sampling
    """
    teams = df["name"].unique()
    with pm.Model() as model:

        # define the hyperprior
        lambda_hp = pm.Uniform("lambda_hp", lower=0, upper=1)

        for team in teams:

            # extract the data for the team
            df_team = df.loc[df["name"] == team]
            score_team = df_team.values[0][2:-1].astype("float")
            score_team = score_team[~np.isnan(score_team)]
            score_team = score_team[:, None]
            # set the prior
            alpha = pm.Deterministic(f"alpha_{team}",
                                     1+pm.Exponential(f"Exp_alpha_{team}",
                                                      lambda_hp))
            beta = pm.Deterministic(f"beta_{team}",
                                    1+pm.Exponential(f"Exp_beta_{team}",
                                                     lambda_hp))

            # normalize the observed score
            score_obs_normalized = (score_team - mean)/(sd**2)+0.5
            too_small_scores = (score_obs_normalized[:, 0] < 0).sum()
            too_large_scores = (score_obs_normalized[:, 0] > 1).sum()
            score_obs_normalized = np.maximum(0, score_obs_normalized)
            score_obs_normalized = np.minimum(1, score_obs_normalized)

            # add this to squeeze the scores between 0.2,0.8 s.t. not
            # too much mass is at the end of the spectrum
            score_obs_normalized_capped = (
                (0.5-score_obs_normalized)/0.5)*cap+score_obs_normalized
            # print(score_obs_normalized)

            if too_small_scores > 0:
                print(team, f": {too_small_scores} scores capped from below")
                print(score_obs_normalized)
            if too_large_scores > 0:
                print(team, f": {too_large_scores} scores capped from above")
                print(score_obs_normalized)

            # model the score
            normalized_score = pm.Beta(f"normalized_score_{team}", alpha=alpha, beta=beta,
                                       observed=score_obs_normalized_capped)
            # reassemble the actual score for later sampling
            score = pm.Deterministic(f"score_{team}", mean + (sd**2)*(
                (-cap*(0.5-normalized_score)/(0.5-cap)+normalized_score-0.5)))

    return model
