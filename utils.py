def compute_points(df, team_id):
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
