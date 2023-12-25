import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from tqdm.auto import tqdm


def extract_all_scores(df: pd.DataFrame) -> np.ndarray:
    """Return all the scores of all the teams in a single 1-dim array

    Args:
        df (pd.DataFrame): Pandas df with columns: "name","points_1", 
        ...,"points_36","total". It contains respectively name of the team,
        points scored by the team in the jth match day, total of the points
        scored by the team

    Returns:
        scores (np.ndarray): 1-dim array containing all the points that have been 
        scored by all the teams.
    """

    teams = df["name"].unique()
    scores = np.empty(0)
    for team in teams:
        df_team = df.loc[df["name"] == team]
        score_team = df_team.values[0][2:-1].astype("float")
        score_team = score_team[~np.isnan(score_team)]
        scores = np.concatenate((scores, score_team))
    return scores


def extract_posteriors(trace: az.InferenceData, model: pm.Model,
                       teams: list) -> dict:
    """ Sample posterior scores for each team

    Args:
        trace (az.InferenceData): trace object to be used to sample posterior
        model (pm.Model): model used to create the trace object
        teams (list): list of the teams

    Returns:
        dict: Dictionary with team name as key and posterior scores as values
    """

    team_scores = {}
    for team in teams:
        post = pm.sample_posterior_predictive(trace, model, [f"score_{team}"])
        post = post.posterior_predictive[f"score_{team}"].values.ravel()
        team_scores[team] = post

    return team_scores


def compute_points(df: pd.DataFrame, team_id: int) -> int:
    """Compute the points obtained by team team_id 

    Args:
        df (pd.DataFrame): Dataframe containing the scores of team
        with team_id=1 and team_id=2
        team_id (int): The team for which the number of points has to be computed

    Returns:
        int: number of points obtained by team team_id
    """

    # check for goals
    team_1_goals = max(0, int(df["team_1_score"]-66) / 4)
    team_2_goals = max(0, int(df["team_2_score"]-66) / 4)

    # check for own goals
    if df["team_1_score"] < 60:
        team_2_goals += 1
    elif df["team_2_score"] < 60:
        team_1_goals += 1

    team_goals = [team_1_goals, team_2_goals]

    # return points for team team_id
    if team_goals[team_id-1] > team_goals[2-team_id]:
        return 3
    elif team_goals[team_id-1] < team_goals[2-team_id]:
        return 0
    else:
        return 1


def simulate_leagues(df_calendar: pd.DataFrame, teams: list, pars: dict,
                     team_scores: dict) -> dict:

    # for every team store the result of every iteration in this dict
    results = {}
    for team in teams:
        results[team] = []

    for _ in tqdm(range(pars["num_league_simulation"])):

        scores_iteration = {}
        standing_iteration = {}

        for team in teams:
            # sample the scores for each team
            scores_iteration[team] = np.random.choice(
                team_scores[team], size=pars["total_match_day"] -
                pars["last_played_match_day"],
                replace=True
            )

        # add the results of every game
        df_calendar["team_1_score"] = df_calendar.apply(
            lambda x: scores_iteration[x["team_1"]][int(x["giornata"])-1 - pars["last_played_match_day"]], axis=1)
        df_calendar["team_2_score"] = df_calendar.apply(
            lambda x: scores_iteration[x["team_2"]][int(x["giornata"])-1 - pars["last_played_match_day"]], axis=1)
        df_calendar["team_1_points"] = df_calendar.apply(
            lambda x: compute_points(x, team_id=1), axis=1)
        df_calendar["team_2_points"] = df_calendar.apply(
            lambda x: compute_points(x, team_id=2), axis=1)

        # sum over the points of every team
        for team in teams:
            points_as_1 = df_calendar.loc[df_calendar["team_1"]
                                          == team, "team_1_points"].sum()
            points_as_2 = df_calendar.loc[df_calendar["team_2"]
                                          == team, "team_2_points"].sum()
            standing_iteration[team] = pars["current_standing"][team] + \
                points_as_1 + points_as_2

        standing_iteration = sorted(
            standing_iteration.items(), key=lambda x: x[1], reverse=True)
        standing_iteration = [team[0] for team in standing_iteration]

        for team in teams:
            results[team].append(1 + standing_iteration.index(team))

    return results
