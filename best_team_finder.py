import pandas as pd
import numpy as np
import time


class BestTeamFinder:
    """
    Uses Dynamic Programming (DP) to find the best fantasy team under constraints:
    - 5 front, 5 back
    - Max 2 per team
    - Max price ≤ 100.8
    - Maximizing Form sum

    Attributes:
        best_filter (pd.DataFrame): Players dataset ["Player", "$", "Form", "Pos", "team"]
        max_price (float): Budget limit.

    Methods:
        find_best_team(): Runs DP optimization to find the best team.
    """

    def __init__(self, best_filter_df, max_price=100):
        """Initializes the optimizer and loads the best available players."""
        self.best_filter = best_filter_df
        self.max_price = max_price
        

        # Standardize position names (avoid capitalization issues)
        self.best_filter["Pos"] = self.best_filter["Pos"].str.lower()

    def find_best_team(self):
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
