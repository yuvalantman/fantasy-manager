import pandas as pd
import itertools
from itertools import combinations


class FantasyOptimizer:
    def __init__(self, my_team, best_filter_path, schedule_path):
        """Initialize and load all required data."""
        if isinstance(my_team, str):  # If given a file path, read the file
            self.my_team = pd.read_csv(my_team)
        else:  # If given a DataFrame, use it directly
            self.my_team = my_team

        self.best_filter = best_filter_path
        self.schedule = pd.read_csv(schedule_path)

        # Ensure column names are correct
        self.schedule.columns = self.schedule.columns.astype(str)

        # Standardize position names (to avoid issues with capitalization)
        self.my_team["Pos"] = self.my_team["Pos"].str.lower()
        self.best_filter["Pos"] = self.best_filter["Pos"].str.lower()
        self.playing_teams_dict = {
            day-1: self.schedule[self.schedule[str(day)] != "-"]["TeamAgg"].tolist()
            for day in range(2, 9)  # Days from 1 to 7
        }
        self.playing_players_dict = {
            day : self.best_filter[self.best_filter["team"].isin(self.playing_teams_dict[day])]["Player"].tolist()
            for day in range(1,8)
        }

    def get_playing_players(self, team_df, day):
        """Returns players from a given team who have a game on the given day."""
        if day not in self.playing_teams_dict.keys:
            return team_df.iloc[0:0]  # Return empty DataFrame
        return team_df[team_df["Player"].isin(self.playing_players_dict[day])]


    def get_weekly_form(self, team_df=None):
        """Computes total weekly form by summing the best possible daily lineups."""
        if team_df is None:
            team_df = self.my_team  # Default to current team if no argument is given

        total_form = 0
        for day in range(1, 8):  # Days 1-7
            playing_players = self.get_playing_players(team_df, day)

            if len(playing_players) > 0:
                back_players = playing_players[playing_players["Pos"] == "back"]
                front_players = playing_players[playing_players["Pos"] == "front"]

                # Select at most 3 from each position
                top_back = back_players.nlargest(3, "Form") if len(back_players) > 0 else pd.DataFrame()
                top_front = front_players.nlargest(3, "Form") if len(front_players) > 0 else pd.DataFrame()

                # Combine the players and select up to 5 total
                top_players = pd.concat([top_back, top_front]).nlargest(5, "Form")

                # Calculate the total form for the day
                daily_form = top_players["Form"].sum()

                total_form += daily_form  # Sum up daily forms

        return total_form

    def find_best_weekly_substitutions(self, extra_salary=0):
        """Finds the best 1-2 substitutions to improve weekly form, considering extra salary."""
        # Ensure salary is handled correctly
        extra_salary = float(extra_salary)

        # Rename salary column to avoid KeyErrors
        self.my_team = self.my_team.rename(columns={"$": "Salary"})
        self.best_filter = self.best_filter.rename(columns={"$": "Salary"})

        current_weekly_form = self.get_weekly_form()
        current_salary = self.my_team["Salary"].sum()
        max_salary = current_salary + extra_salary

        available_players = self.best_filter[~self.best_filter["Player"].isin(self.my_team["Player"])]
        available_players = available_players.sort_values(by="Form", ascending=False)

        best_team = self.my_team.copy()
        best_weekly_form = current_weekly_form
        best_salary = current_salary
        best_out, best_in = [], []

        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(self.my_team.itertuples(index=False, name="PlayerTuple"), num_swaps):
                out_list = list(out_players)
                out_dicts = [p._asdict() for p in out_list]

                available_salary = sum(float(p["Salary"]) for p in out_dicts) + extra_salary
                needed_pos = [p["Pos"] for p in out_dicts]

                valid_replacements = available_players[
                    (available_players["Pos"].isin(needed_pos)) | 
                    (available_players["Pos"].value_counts().get("front", 0) < 5) | 
                    (available_players["Pos"].value_counts().get("back", 0) < 5)
                ].sort_values(by="Form", ascending=False).head(20)

                for in_players in itertools.combinations(valid_replacements.itertuples(index=False, name="PlayerTuple"), num_swaps):
                    in_list = list(in_players)
                    if not in_list:
                        continue  

                    in_dicts = [p._asdict() for p in in_list]

                    new_team = self.my_team[~self.my_team["Player"].isin([p["Player"] for p in out_dicts])].copy()
                    new_team = pd.concat([new_team, pd.DataFrame(in_dicts, columns=self.my_team.columns)], ignore_index=True)

                    new_form = self.get_weekly_form(team_df=new_team)
                    new_salary = new_team["Salary"].sum()

                    if new_salary > max_salary:
                        continue  

                    front_count = new_team["Pos"].value_counts().get("front", 0)
                    back_count = new_team["Pos"].value_counts().get("back", 0)

                    if front_count != 5 or back_count != 5:
                        continue  

                    if new_form > best_weekly_form or (new_form == best_weekly_form and new_salary < best_salary):
                        best_team = new_team.copy()
                        best_weekly_form = new_form
                        best_salary = new_salary
                        best_out = [p["Player"] for p in out_dicts]
                        best_in = [p["Player"] for p in in_dicts]

        print(f"🔵 Current Weekly Form: {current_weekly_form}, Salary: {current_salary}")
        print(f"🟢 New Weekly Form: {best_weekly_form}, Salary: {best_salary}")
        if best_out and best_in:
            print(f"🔄 Substitutions Made:")
            print(f"❌ Out: {', '.join(best_out)}")
            print(f"✅ In: {', '.join(best_in)}")
        else:
            print("✅ No substitutions made (no possible improvement within salary cap).")


        return best_team, best_weekly_form.round(2), best_salary.round(1), best_out, best_in
    

    def find_best_total_substitutions(self, extra_salary=0):
        """ 
        Finds the best 1 or 2 substitutions that improve total form 
        while respecting salary cap and position balance.
        
        Parameters:
            df (pd.DataFrame): Full list of good players.
            my_team (pd.DataFrame): Current 10-player team.
        
        Returns:
            pd.DataFrame: The new best team after substitutions.
        """

        # Compute current total form and salary
        current_form = self.my_team["Form"].sum()
        current_salary = self.my_team["$"].sum()

        # Only consider top 50 players by form instead of all
        available_players = self.best_filter[~self.best_filter["Player"].isin(self.my_team["Player"])].copy()
        available_players = available_players.sort_values(by="Form", ascending=False)

        best_team = self.my_team.copy()
        best_form = current_form
        best_salary = current_salary
        best_out, best_in = [], []

        # Try 1-player and 2-player swaps
        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(self.my_team.itertuples(index=False, name=None), num_swaps):
                out_list = list(out_players)

                # Compute available salary (sum of removed players)
                available_salary = sum(float(player[2]) for player in out_list)

                # Get positions needed after removing these players
                needed_pos = [player[5] for player in out_list]

                # Filter replacements by position and salary
                valid_replacements = available_players[
                    (available_players["Pos"].isin(needed_pos)) & 
                    (available_players["$"] <= available_salary + extra_salary)
                ].sort_values(by="Form", ascending=False).head(20)  # Further limit to top 20

                # Try all possible replacement combinations
                for in_players in itertools.combinations(valid_replacements.itertuples(index=False, name=None), num_swaps):
                    in_list = list(in_players)

                    # Ensure 5 front / 5 back balance
                    new_team = self.my_team[~self.my_team["Player"].isin([p[1] for p in out_list])].copy()
                    new_team = pd.concat([new_team, pd.DataFrame(in_list, columns=self.my_team.columns)], ignore_index=True)

                    new_form = new_team["Form"].sum()
                    new_salary = new_team["$"].sum()

                    if (new_team["Pos"].value_counts().get("front", 0) == 5 and 
                        new_team["Pos"].value_counts().get("back", 0) == 5 and
                        new_salary <= current_salary + extra_salary):  # Ensure salary cap is respected

                        if new_form > best_form:
                            best_team = new_team.copy()
                            best_form = new_form
                            best_salary = new_salary
                            best_out = [p[1] for p in out_list]
                            best_in = [p[1] for p in in_list]

        # Print Summary
        print(f"🔵 Current Team - Form: {current_form}, Salary: {current_salary}")
        print(f"🟢 New Team - Form: {best_form}, Salary: {best_salary}")
        if best_out and best_in:
            print(f"🔄 Substitutions Made:")
            print(f"❌ Out: {', '.join(best_out)}")
            print(f"✅ In: {', '.join(best_in)}")
        else:
            print("✅ No substitutions made (no possible improvement within salary cap).")
        return best_team, best_form.round(2), best_salary.round(1), best_out, best_in



