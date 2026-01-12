import pandas as pd
import numpy as np
import time


class BestTeamFinder:
    """
    Uses Dynamic Programming (DP) to find the best fantasy team under constraints:
    - 5 front, 5 back
    - Max 2 per team
    - Max price ≤ 100
    - Maximizing Form sum

    Attributes:
        best_filter (pd.DataFrame): Players dataset ["Player", "$", "Form", "Pos", "team"]
        max_price (float): Budget limit.

    Methods:
        find_best_team(): Runs DP optimization to find the best team.
    """

    # def __init__(self, best_filter_df, max_price=100):
    #     """Initializes the optimizer and loads the best available players."""
    #     self.best_filter = best_filter_df
    #     self.max_price = max_price
        

    #     # Standardize position names (avoid capitalization issues)
    #     self.best_filter["Pos"] = self.best_filter["Pos"].str.lower()

    def __init__(self, players_df, max_price=100):
        """Initializes the optimizer with the player dataset."""
        self.players_df = players_df.copy()
        self.players_df["Pos"] = self.players_df["Pos"].str.lower()
        self.max_price = max_price
        self.max_price_int = int(self.max_price * 10)

    def _knapsack_dp(self, players_df, k_players, score_column='Form'):
        """
        Performs a dynamic programming calculation for a single position group.
        
        Returns:
            dp (np.array): DP table where dp[k][w] is the max score for k players at cost w.
            choices (dict): Dictionary for backtracking to reconstruct the team.
        """
        n = len(players_df)
        dp = np.full((k_players + 1, self.max_price_int + 1), -1.0)
        dp[0, :] = 0
        choices = {}

        for i, player in players_df.iterrows():
            cost = int(player["$"] * 10)
            score = player[score_column]
            
            for k in range(k_players, 0, -1):
                for w in range(self.max_price_int, cost - 1, -1):
                    if dp[k - 1, w - cost] != -1:
                        new_score = dp[k - 1, w - cost] + score
                        if new_score > dp[k, w]:
                            dp[k, w] = new_score
                            choices[(k, w)] = (player['Player'], k - 1, w - cost)
        return dp, choices

    def _backtrack(self, players_df, choices, k, w):
        """Reconstructs the list of players from the choices table."""
        team = []
        while (k, w) in choices:
            player_name, prev_k, prev_w = choices[(k, w)]
            team.append(player_name)
            k, w = prev_k, prev_w
        
        return players_df[players_df['Player'].isin(team)]

    def _repair_team_constraint(self, team_df, available_pool, score_column='Form'):
        """Heuristic to fix the 'max 2 players per team' constraint."""
        repaired_team = team_df.copy()
        
        while True:
            team_counts = repaired_team['team'].value_counts()
            violating_teams = team_counts[team_counts > 2]

            if violating_teams.empty:
                break

            # Fix the first violating team found
            team_to_fix = violating_teams.index[0]
            
            # Identify players to remove (the ones with the lowest score)
            violating_players = repaired_team[repaired_team['team'] == team_to_fix].sort_values(by=score_column, ascending=True)
            num_to_remove = len(violating_players) - 2
            players_to_remove = violating_players.head(num_to_remove)

            repaired_team = repaired_team[~repaired_team['Player'].isin(players_to_remove['Player'])]
            
            # Find best replacements
            current_salary = repaired_team['$'].sum()
            salary_cap_for_replacements = self.max_price - current_salary
            
            # Find candidates that don't violate existing team counts
            valid_teams = repaired_team['team'].value_counts()[repaired_team['team'].value_counts() < 2].index
            
            for _, player_out in players_to_remove.iterrows():
                pos_needed = player_out['Pos']
                
                candidates = available_pool[
                    (~available_pool['Player'].isin(repaired_team['Player'])) &
                    (available_pool['Pos'] == pos_needed) &
                    (available_pool['$'] <= salary_cap_for_replacements) &
                    (available_pool['team'].isin(valid_teams))
                ].sort_values(by=score_column, ascending=False)
                
                if not candidates.empty:
                    best_replacement = candidates.iloc[[0]]
                    repaired_team = pd.concat([repaired_team, best_replacement], ignore_index=True)
                    salary_cap_for_replacements -= best_replacement['$'].iloc[0]

        return repaired_team

    def find_best_team(self, score_column='Form'):
        """
        Finds the single best team by running DP on fronts and backs separately
        and combining the results.
        """
        fronts = self.players_df[self.players_df['Pos'] == 'front'].sort_values(by=score_column, ascending=False).reset_index(drop=True)
        backs = self.players_df[self.players_df['Pos'] == 'back'].sort_values(by=score_column, ascending=False).reset_index(drop=True)

        dp_front, choices_front = self._knapsack_dp(fronts, 5, score_column)
        dp_back, choices_back = self._knapsack_dp(backs, 5, score_column)

        best_score = -1
        best_split = (0, 0)

        for w_front in range(self.max_price_int + 1):
            w_back = self.max_price_int - w_front
            if dp_front[5, w_front] != -1 and dp_back[5, w_back] != -1:
                total_score = dp_front[5, w_front] + dp_back[5, w_back]
                if total_score > best_score:
                    best_score = total_score
                    best_split = (w_front, w_back)

        if best_score == -1:
            return None # No valid team found

        w_front, w_back = best_split
        best_fronts = self._backtrack(fronts, choices_front, 5, w_front)
        best_backs = self._backtrack(backs, choices_back, 5, w_back)
        
        optimal_team = pd.concat([best_fronts, best_backs], ignore_index=True)
        
        # Repair team constraint
        available_pool = self.players_df[~self.players_df['Player'].isin(optimal_team['Player'])]
        repaired_team = self._repair_team_constraint(optimal_team, available_pool, score_column)
        
        return repaired_team

    def _calculate_true_biweekly_score(self, team_df, playing_teams_dict):
        """
        Calculates the true score of a team over a 14-day schedule based on daily optimal lineups.
        """
        total_score = 0
        team_players_by_pos = {
            'front': team_df[team_df['Pos'] == 'front'],
            'back': team_df[team_df['Pos'] == 'back']
        }

        for day in range(2, 16):
            playing_teams_today = playing_teams_dict.get(day, [])
            if not playing_teams_today:
                continue

            playing_fronts = team_players_by_pos['front'][team_players_by_pos['front']['team'].isin(playing_teams_today)]
            playing_backs = team_players_by_pos['back'][team_players_by_pos['back']['team'].isin(playing_teams_today)]

            # Get top 3 from each position that are playing
            top_fronts = playing_fronts.nlargest(3, 'Form')
            top_backs = playing_backs.nlargest(3, 'Form')

            # Combine and get top 5 overall for the day
            daily_lineup_pool = pd.concat([top_fronts, top_backs])
            top_5_for_day = daily_lineup_pool.nlargest(5, 'Form')
            
            daily_score = top_5_for_day['Form'].sum()
            total_score += daily_score
            
        return total_score

    def find_best_biweekly_team(self, schedule_df, top_n=5, candidate_pool_size=500):
        """
        Finds top N teams by generating a large pool of candidates and evaluating their
        true bi-weekly score.
        """
        print(f"🚀 Generating a pool of up to {candidate_pool_size} candidate teams...")
        
        # === Step 1: Generate Candidate Teams ===
        candidate_teams = []
        fronts = self.players_df[self.players_df['Pos'] == 'front'].sort_values(by='Form', ascending=False).reset_index(drop=True)
        backs = self.players_df[self.players_df['Pos'] == 'back'].sort_values(by='Form', ascending=False).reset_index(drop=True)

        dp_front, choices_front = self._knapsack_dp(fronts, 5, score_column='Form')
        dp_back, choices_back = self._knapsack_dp(backs, 5, score_column='Form')
        
        all_splits = []
        for w_front in range(self.max_price_int + 1):
            w_back = self.max_price_int - w_front
            if w_back >= 0 and dp_front[5, w_front] > -1 and dp_back[5, w_back] > -1:
                total_score = dp_front[5, w_front] + dp_back[5, w_back]
                all_splits.append({'score': total_score, 'w_front': w_front, 'w_back': w_back})
        
        all_splits.sort(key=lambda x: x['score'], reverse=True)
        
        seen_teams = set()
        for split in all_splits[:candidate_pool_size]:
            w_front, w_back = split['w_front'], split['w_back']
            team_front = self._backtrack(fronts, choices_front, 5, w_front)
            team_back = self._backtrack(backs, choices_back, 5, w_back)
            
            if len(team_front) == 5 and len(team_back) == 5:
                team_df = pd.concat([team_front, team_back], ignore_index=True)
                repaired_team = self._repair_team_constraint(team_df, self.players_df, score_column='Form')
                
                team_key = tuple(sorted(repaired_team['Player'].tolist()))
                if team_key not in seen_teams:
                    candidate_teams.append(repaired_team)
                    seen_teams.add(team_key)

        print(f"✅ Generated {len(candidate_teams)} unique candidate teams.")
        if not candidate_teams:
            return []

        # === Step 2: Evaluate Each Candidate Team ===
        print(f"🔬 Evaluating {len(candidate_teams)} teams against the bi-weekly schedule...")
        
        playing_teams_dict = {
            day: schedule_df[schedule_df[str(day)] != "-"]["TeamAgg"].tolist()
            for day in range(1, 15) if str(day) in schedule_df.columns
        }

        evaluated_teams = []
        for i, team_df in enumerate(candidate_teams):
            if (i + 1) % 100 == 0:
                print(f"  ...evaluating team {i+1}/{len(candidate_teams)}")
            
            true_score = self._calculate_true_biweekly_score(team_df, playing_teams_dict)
            evaluated_teams.append({'score': true_score, 'team': team_df})

        # === Step 3: Sort and Return Top N ===
        print("🏆 Sorting teams by true bi-weekly score...")
        evaluated_teams.sort(key=lambda x: x['score'], reverse=True)
        
        return [{
            'score': item['score'],
            'team': item['team'],
            'total_price': item['team']["$"].sum()
        } for item in evaluated_teams[:top_n]]
    
    
    
    
    
    
    
    
    
    
    def find_best_team2(self):
        """Finds the best fantasy team using DP optimization."""
        best_filter = self.best_filter.sort_values(by="Form", ascending=False).reset_index(drop=True)

        n = len(best_filter)
        max_w = int(self.max_price * 10)  # Convert price to an integer

        # **Step 2: DP Table Initialization**
        dp = np.zeros((2, max_w + 1, 11))  # 2 layers for space optimization

        # **Step 3: Backtracking Information**
        choices = np.full((n, max_w + 1, 11), -1)  # Store decisions

        # **Step 4: Iterate Over Players**
        for i in range(n):
            player = best_filter.iloc[i]
            cost = int(player["$"] * 10)  # Convert to integer for DP
            form = player["Form"]
            pos = player["Pos"]
            team = player["team"]

            curr = i % 2  # Current layer
            prev = 1 - curr  # Previous layer

            # **Step 5: DP Transition**
            for w in range(max_w + 1):
                for k in range(11):
                    # **Option 1: Do not pick this player**
                    dp[curr][w][k] = dp[prev][w][k]

                    # **Option 2: Pick this player (if valid)**
                    if k > 0 and w >= cost:
                        new_form = dp[prev][w - cost][k - 1] + form

                        # **Check Position & Team Constraints**
                        valid = True
                        if k <= 5 and pos == "back":  # More than 5 backs?
                            valid = False
                        if k > 5 and pos == "front":  # More than 5 fronts?
                            valid = False

                        if valid and new_form > dp[curr][w][k]:
                            dp[curr][w][k] = new_form
                            choices[i][w][k] = w - cost  # Store path

        # **Step 6: Backtrack to Find the Team**
        best_w, best_k = max_w, 10
        best_form = dp[(n - 1) % 2][best_w][best_k]

        if best_form == 0:
            return None  # No valid team found

        # **Find Selected Players**
        selected_players = []
        i = n - 1

        while best_k > 0 and i >= 0:
            if choices[i][best_w][best_k] != -1:
                selected_players.append(best_filter.iloc[i])
                best_w = choices[i][best_w][best_k]
                best_k -= 1
            i -= 1

        return pd.DataFrame(selected_players)


    def find_best_team1(self):
        """Finds the best fantasy team using DP optimization."""
        best_filter = self.best_filter.sort_values(by="Form", ascending=False).reset_index(drop=True)
        n = len(best_filter)
        max_w = int(self.max_price * 10)

        dp = np.zeros((2, max_w + 1, 11))
        choices = np.full((n, max_w + 1, 11), -1)
        pos_tracker = np.full((2, max_w + 1, 11), None)

        # Initialize base case
        for w in range(max_w + 1):
            pos_tracker[0][w][0] = (0, 0)

        for i in range(n):
            player = best_filter.iloc[i]
            cost = int(player["$"] * 10)
            form = player["Form"]
            pos = player["Pos"]

            curr = i % 2
            prev = 1 - curr

            for w in range(max_w + 1):
                for k in range(11):
                    # Option 1: don't pick player
                    dp[curr][w][k] = dp[prev][w][k]
                    choices[i][w][k] = -1
                    pos_tracker[curr][w][k] = pos_tracker[prev][w][k]

                    # Option 2: try to pick player
                    if k > 0 and w >= cost:
                        prev_bf = pos_tracker[prev][w - cost][k - 1]
                        if prev_bf is None:
                            continue
                        backs, fronts = prev_bf

                        if pos == "back" and backs >= 5:
                            continue
                        if pos == "front" and fronts >= 5:
                            continue

                        new_form = dp[prev][w - cost][k - 1] + form
                        if new_form > dp[curr][w][k]:
                            dp[curr][w][k] = new_form
                            choices[i][w][k] = w - cost
                            if pos == "back":
                                pos_tracker[curr][w][k] = (backs + 1, fronts)
                            else:
                                pos_tracker[curr][w][k] = (backs, fronts + 1)

        # Backtrack
        best_w, best_k = max_w, 10
        best_form = dp[(n - 1) % 2][best_w][best_k]
        if best_form == 0:
            return None

        selected_players = []
        team_counts = {}
        i = n - 1

        while best_k > 0 and i >= 0:
            prev_w = choices[i][best_w][best_k]
            if prev_w != -1:
                player = best_filter.iloc[i]
                team = player["team"]

                # Enforce max 2 per team
                if team_counts.get(team, 0) >= 2:
                    i -= 1
                    continue

                selected_players.append(player)
                team_counts[team] = team_counts.get(team, 0) + 1

                best_w = prev_w
                best_k -= 1
            i -= 1

        return pd.DataFrame(selected_players)
