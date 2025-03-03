import pandas as pd
import itertools
from itertools import combinations
import heapq

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
        self.playing_teams_dict19 = {
            day-1: self.schedule[self.schedule[str(day)] != "-"]["TeamAgg"].tolist()
            for day in range(2, 9)  # Days from 1 to 7
        }
        self.playing_teams_dict = {
            1: ['ATL', 'CHA', 'DET', 'MIA', 'PHI', 'POR', 'SAC', 'UTA', 'WAS', 'GSW', 'MEM', 'HOU', 'OKC', 'DAL'], 
            2: ['ATL', 'BKN', 'CHI', 'IND', 'LAC', 'MIN', 'PHI', 'HOU', 'TOR', 'ORL', 'MIL', 'GSW', 'NYK', 'CLE', 'SAS', 'PHX', 'NOP', 'LAL'],
            3: ['POR', 'BOS', 'MIN', 'CHA', 'MIA', 'CLE', 'UTA', 'WAS', 'DAL', 'MIL', 'SAC', 'DEN', 'OKC', 'MEM', 'DET', 'LAC'],
            4: ['CHI', 'ORL', 'IND', 'ATL', 'PHI', 'BOS', 'GSW', 'BKN', 'HOU', 'NOP', 'NYK', 'LAL'],
            5: ['CLE', 'CHA', 'MEM', 'DAL', 'UTA', 'TOR', 'MIN', 'MIA', 'POR', 'OKC', 'PHX', 'DEN', 'SAS', 'SAC', 'NYK', 'LAC'],
            6: ['BKN', 'CHA', 'NOP', 'HOU', 'IND', 'ATL', 'WAS', 'TOR', 'CHI', 'MIA', 'ORL', 'MIL', 'LAL', 'BOS', 'DET', 'GSW'],
            7: ['DEN', 'OKC', 'PHX', 'DAL', 'MEM', 'NOP', 'PHI', 'UTA', 'CLE', 'MIL', 'SAS', 'MIN', 'DET', 'POR', 'SAC', 'LAC']

        }
        print(self.playing_teams_dict)
        self.playing_players_dict = {
            day : self.best_filter[self.best_filter["team"].isin(self.playing_teams_dict[day])]["Player"].tolist()
            for day in range(1,8)
        }

    def get_playing_players(self, team_df, day):
        """Returns players from a given team who have a game on the given day."""
        if day not in self.playing_teams_dict.keys():
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
    
    def print_weekly_form(self,team_df = None):
        if team_df is None:
            team_df = self.my_team  # Default to current team if no argument is given

        weekly_sched = {}
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
                daily_form = top_players["Form"].sum()
                weekly_sched[day] = top_players[["Player", "Form"]].to_dict(orient="records")  # Convert DataFrame to list of dicts
                weekly_sched[day].append({"Daily Total": daily_form.round(1)})
            else:
                weekly_sched[day] = []
        return weekly_sched

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
                    (available_players["Pos"].isin(needed_pos)) & available_players["Salary"] <= available_salary
                ].sort_values(by="Form", ascending=False).head(20)
                if (len(valid_replacements) < num_swaps):
                    continue

                for in_players in itertools.combinations(valid_replacements.itertuples(index=False, name="PlayerTuple"), num_swaps):
                    in_list = list(in_players)
                    if not in_list:
                        continue  

                    in_dicts = [p._asdict() for p in in_list]

                    new_team = self.my_team[~self.my_team["Player"].isin([p["Player"] for p in out_dicts])].copy()
                    new_team = pd.concat([new_team, pd.DataFrame(in_dicts, columns=self.my_team.columns)], ignore_index=True)

                    new_salary = new_team["Salary"].sum()

                    if new_salary > max_salary:
                        continue  

                    front_count = new_team["Pos"].value_counts().get("front", 0)
                    back_count = new_team["Pos"].value_counts().get("back", 0)

                    if front_count != 5 or back_count != 5:
                        continue  
                    new_form = self.get_weekly_form(team_df=new_team)

                    if new_form > best_weekly_form or (new_form == best_weekly_form and new_salary < best_salary):
                        best_team = new_team.copy()
                        best_weekly_form = new_form
                        best_salary = new_salary
                        best_out = [p["Player"] for p in out_dicts]
                        best_in = [p["Player"] for p in in_dicts]
        weekly_sched = self.print_weekly_form(best_team)
        print(f"🔵 Current Weekly Form: {current_weekly_form}, Salary: {current_salary}")
        print(f"🟢 New Weekly Form: {best_weekly_form}, Salary: {best_salary}")
        if best_out and best_in:
            print(f"🔄 Substitutions Made:")
            print(f"❌ Out: {', '.join(best_out)}")
            print(f"✅ In: {', '.join(best_in)}")
        else:
            print("✅ No substitutions made (no possible improvement within salary cap).")


        return best_team, best_weekly_form.round(2), best_salary.round(1), best_out, best_in, weekly_sched
    

    def find_best_total_substitutions(self, extra_salary=0, top_n=5):
        """Finds the best 1 or 2 substitutions that improve total form while respecting salary cap and position balance."""

        print("\n🔍 DEBUG: Starting Optimized Best Total Substitutions...\n")

        # Compute current total form and salary
        current_form = self.my_team["Form"].sum()
        current_salary = self.my_team["$"].sum()
        max_salary = current_salary + extra_salary

        print(f"🔵 Current Team Form: {current_form}, Salary: {current_salary}, Max Salary: {max_salary}\n")

        # Ensure correct column naming
        self.my_team = self.my_team.rename(columns={"$": "Salary"})
        self.best_filter = self.best_filter.rename(columns={"$": "Salary"})
        self.my_team["Salary"] = self.my_team["Salary"].astype(float)
        self.best_filter["Salary"] = self.best_filter["Salary"].astype(float)

        # Sort my team by Form (ascending) → Remove weak players first
        self.my_team = self.my_team.sort_values(by="Form", ascending=True)

        # Sort available players by Form (descending) → Add strong players first
        available_players = self.best_filter[~self.best_filter["Player"].isin(self.my_team["Player"])].copy()
        available_players = available_players.sort_values(by="Form", ascending=False)

        print(f"📌 Available Players Before Filtering: {len(available_players)}\n")

        # **Step 1: Precompute all available player swap combinations**
        available_combos = {
            "1": [],  # 1-player swaps
            "FF": [], # 2-player front swaps
            "BB": [], # 2-player back swaps
            "FB": []  # 2-player mixed swaps
        }

        for num_swaps in [1, 2]:
            for in_players in itertools.combinations(available_players.itertuples(index=False, name="PlayerTuple"), num_swaps):
                in_df = pd.DataFrame(in_players, columns=self.my_team.columns)
                form_sum = in_df["Form"].sum()
                salary_sum = in_df["Salary"].sum()

                # Categorize by position
                if num_swaps == 1:
                    key = "1"
                else:
                    pos_tuple = tuple(sorted(p.Pos for p in in_players))
                    if pos_tuple == ("front", "front"):
                        key = "FF"
                    elif pos_tuple == ("back", "back"):
                        key = "BB"
                    else:
                        key = "FB"

                available_combos[key].append((form_sum, salary_sum, in_df))

        # **Step 2: Sort each category by salary (ascending)**
        for key in available_combos:
            available_combos[key].sort(key=lambda x: x[1])  # Sort by salary sum (low to high)

        # **Step 3: Track the top N best swaps using a Min Heap**
        top_swaps = []  # Will store (form_gain, new_team, new_form, new_salary, out_list, in_list)
        min_top_form = float('-inf')  # Track the smallest form gain in the heap

        # **Step 4: Iterate through all possible user team swap combinations (1 and 2 swaps)**
        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(self.my_team.itertuples(index=False, name="PlayerTuple"), num_swaps):
                out_list = list(out_players)
                available_salary = sum(float(player.Salary) for player in out_list) + extra_salary

                # Get correct precomputed swap list based on position combination
                if num_swaps == 1:
                    swap_list = available_combos["1"]
                else:
                    pos_tuple = tuple(sorted(p.Pos for p in out_list))
                    if pos_tuple == ("front", "front"):
                        swap_list = available_combos["FF"]
                    elif pos_tuple == ("back", "back"):
                        swap_list = available_combos["BB"]
                    else:
                        swap_list = available_combos["FB"]

                # **Step 5: Filter valid swaps by salary**
                valid_swaps = [swap for swap in swap_list if swap[1] <= available_salary]
                if not valid_swaps:
                    continue  # Skip if no valid swaps exist

                # **Step 6: Pick the best swap (max form sum)**
                best_form_swap = max(valid_swaps, key=lambda x: x[0])  # Get max form sum
                form_sum, salary_sum, in_df = best_form_swap

                # **Step 7: Compute form gain EARLY (before creating new team)**
                form_gain = form_sum - sum(player.Form for player in out_list)

                # If form gain is NOT better than `min_top_form`, skip early
                if len(top_swaps) >= top_n and form_gain <= min_top_form:
                    continue  # 🔥 No need to create new team, skip it

                # **Step 8: Create New Team**
                new_team = self.my_team[~self.my_team["Player"].isin([p.Player for p in out_list])].copy()
                new_team = pd.concat([new_team, in_df], ignore_index=True)

                new_team["Salary"] = new_team["Salary"].astype(float)
                new_form = new_team["Form"].sum()
                new_salary = new_team["Salary"].sum()

                # **Step 9: Track the top N best swaps**
                if len(top_swaps) < top_n:
                    heapq.heappush(top_swaps, (
                        round(float(form_gain), 1),  
                        round(float(new_form), 1),  
                        round(float(new_salary), 1),  
                        [p.Player for p in out_list], 
                        list(in_df["Player"]),
                        new_team.to_dict(orient="records")  # ✅ Move to the end
                    ))
                else:
                    heapq.heappushpop(top_swaps, (
                        round(float(form_gain), 1),  
                        round(float(new_form), 1),  
                        round(float(new_salary), 1),  
                        [p.Player for p in out_list], 
                        list(in_df["Player"]),
                        new_team.to_dict(orient="records")  # ✅ Move to the end
                    ))
                # **Update min_top_form**
                min_top_form = top_swaps[0][0]  # The smallest form gain in the heap


        # **Step 10: Get the top N best swaps (sorted by highest form gain)**
        top_swaps.sort(reverse=True, key=lambda x: x[0])  # Sort by highest form gain

        return top_swaps  # Returns a list of top swaps instead of just the best one







