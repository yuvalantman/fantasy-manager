import pandas as pd
import itertools
from itertools import combinations
import heapq
import copy
from numba import jit
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
from multiprocessing import Pool
import logging
import bisect

class FantasyOptimizer:
    def __init__(self, my_team, best_filter_path, schedule_week1: dict, schedule_week2: dict | None = None):
        """Initialize and load all required data."""
        if isinstance(my_team, str):  # If given a file path, read the file
            self.my_team = pd.read_csv(my_team)
        else:  # If given a DataFrame, use it directly
            self.my_team = my_team.copy()

        self.best_filter = best_filter_path.copy()

        # Standardize position names (to avoid issues with capitalization)
        self.my_team["Pos"] = self.my_team["Pos"].str.lower()
        self.best_filter["Pos"] = self.best_filter["Pos"].str.lower()
        self.playing_teams_dict = {
            day: schedule_week1.get(day, [])
            for day in range(1, 8)
        }

        self.playing_teams_second_dict = (
            {day: schedule_week2.get(day, []) for day in range(1, 8)}
            if schedule_week2 is not None
            else {}
        )
        
        self.playing_players_dict = {
            day : self.best_filter[self.best_filter["team"].isin(self.playing_teams_dict[day])]["Player"].tolist()
            for day in range(1,8)
        }
        # second week
        self.playing_players_second_dict = {
            day: self.best_filter[self.best_filter["team"].isin(self.playing_teams_second_dict.get(day, []))]["Player"].tolist()
            for day in range(1, 8)
        } if self.playing_teams_second_dict else {}

        self.playing_players_lookup = {
            (player, day): True
            for day, players in self.playing_players_dict.items()
            for player in players
        }

        self.playing_players_second_lookup = {
            (player, day): True
            for day, players in self.playing_players_second_dict.items()
            for player in players
        }

        self.form_lookup = {
            (row["Player"], row["team"]): row["Form"] for _, row in self.best_filter.iterrows()
        }
        self.best_filter_front = self.best_filter[self.best_filter["Pos"] == "front"]
        self.best_filter_back = self.best_filter[self.best_filter["Pos"] == "back"]

        # 🔧 Build pool matrices on copies that already have 'Salary'
        bf_for_pool = self.best_filter.copy()
        if "$" in bf_for_pool.columns and "Salary" not in bf_for_pool.columns:
            bf_for_pool = bf_for_pool.rename(columns={"$": "Salary"})
        bf_for_pool["Salary"] = bf_for_pool["Salary"].astype(float)

        self.pool_matrix_week1 = self.build_player_day_matrix(bf_for_pool)

        if schedule_week2 is not None:
            bf_for_pool2 = bf_for_pool.copy()
            self.pool_matrix_week2 = self.build_player_day_matrix_second(bf_for_pool2)
        else:
            self.pool_matrix_week2 = None

    
    def build_player_day_matrix(self, df=None):
        """
        Adds columns Day1–Day7 to the given DataFrame (default: full pool),
        where each column represents the player's form if they're playing that day.
        """
        if df is None:
            df = self.best_filter.copy()

        df["Form"] = df["Form"].astype(float)
        df["Pos"] = df["Pos"].str.lower()

        for day in range(1, 8):
            playing_teams = self.playing_teams_dict.get(day, [])
            df[f"Day{day}"] = df.apply(
                lambda row: row["Form"] if row["team"] in playing_teams else 0.0,
                axis=1
            )
        print("✅ Player-day matrix built with Day1–Day7 columns.")
        return df
    
    def build_day_lineup(self, team_matrix):
        day_lineup = {}

        for day in range(1, 8):
            day_col = f"Day{day}"
            lineup = {
                "players": set(),
                "form_total": 0.0,
                "front": [],
                "back": [],
                "total": []
            }
            df_day = team_matrix[team_matrix[day_col] > 0.0]
            for _, row in df_day.iterrows():
                entry = (row[day_col], row["Player"], row["Pos"])
                lineup[row["Pos"]].append(entry)
                lineup["total"].append(entry)
                lineup["players"].add(row["Player"])

            top_front = heapq.nlargest(3, lineup["front"], key=lambda x: x[0])
            top_back = heapq.nlargest(3, lineup["back"], key=lambda x: x[0])
            top_all = heapq.nlargest(5, top_front + top_back, key=lambda x: x[0])

            lineup["front"] = [p for p in top_all if p[2] == "front"]
            lineup["back"] = [p for p in top_all if p[2] == "back"]
            lineup["total"] = top_all
            lineup["form_total"] = sum(p[0] for p in top_all)
            day_lineup[day] = lineup

        print("✅ Initial daily lineups built.")
        return day_lineup
    
    def update_day_lineup_remove(self, day_lineup, player_row):
        name = player_row["Player"]
        pos = player_row["Pos"]
        #print(f"🔻 Removing {name} from lineup")
        for day in range(1, 8):
            day_col = f"Day{day}"
            form = player_row[day_col]
            if form == 0:
                continue

            day_data = day_lineup[day]
            if name not in day_data["players"]:
                continue

            
            day_data["players"].remove(name)
            day_data[pos] = [p for p in day_data[pos] if p[1] != name]
            day_data["total"] = [p for p in day_data["total"] if p[1] != name]

            top_front = heapq.nlargest(3, day_data["front"], key=lambda x: x[0])
            top_back = heapq.nlargest(3, day_data["back"], key=lambda x: x[0])
            top_all = heapq.nlargest(5, top_front + top_back, key=lambda x: x[0])

            day_data["front"] = [p for p in top_all if p[2] == "front"]
            day_data["back"] = [p for p in top_all if p[2] == "back"]
            day_data["total"] = top_all
            day_data["form_total"] = sum(p[0] for p in top_all)
    
    def update_day_lineup_add(self, day_lineup, player_row):
        name = player_row["Player"]
        pos = player_row["Pos"]
        #print(f"🔺 Adding {name} to lineup")
        for day in range(1, 8):
            day_col = f"Day{day}"
            form = player_row[day_col]
            if form == 0:
                continue

            day_data = day_lineup[day]
            # Add the new player to the existing pool
            updated_front = day_data["front"] + [(form, name, pos)] if pos == "front" else day_data["front"]
            updated_back = day_data["back"] + [(form, name, pos)] if pos == "back" else day_data["back"]

            # Get top 3 from each
            top_front = heapq.nlargest(3, updated_front, key=lambda x: x[0])
            top_back = heapq.nlargest(3, updated_back, key=lambda x: x[0])

            # From the top 3 front and back, get top 5 overall
            top_all = heapq.nlargest(5, top_front + top_back, key=lambda x: x[0])

            day_data["players"].add(name)
            day_data["front"] = [p for p in top_all if p[2] == "front"]
            day_data["back"] = [p for p in top_all if p[2] == "back"]
            day_data["total"] = top_all
            day_data["form_total"] = sum(p[0] for p in top_all)
            
        return day_lineup
    
    def clone_lineup(self, lineup):
        return {
            day: {
                "players": set(data["players"]),
                "form_total": data["form_total"],
                "front": list(data["front"]),
                "back": list(data["back"]),
                "total": list(data["total"])
            } for day, data in lineup.items()
        }
    
    def evaluate_swap_fast(self, in_players, base_lineup):
        team_lineup = self.clone_lineup(base_lineup)
        # Only add in_players now — out_players are assumed already removed
        for row in in_players:
            day_lineup = self.update_day_lineup_add(team_lineup, row)

        total_form = sum(day_lineup[day]["form_total"] for day in range(1, 8))
        #print(f"📊 Weekly form after swap: {total_form:.2f}")
        return total_form

    def print_weekly_form(self, team_df=None):
        if team_df is None:
            team_df = self.my_team  # Default to current team if not provided

        # Create player-day matrix (adds Day1 to Day7 columns)
        team_matrix = self.build_player_day_matrix(team_df)

        # Build daily best lineups (5 best players per day with max 3 front and 3 back)
        day_lineups = self.build_day_lineup(team_matrix)

        weekly_sched = {}

        for day in range(1, 8):
            players = day_lineups[day]["total"]
            form_sum = day_lineups[day]["form_total"]

            # Format each player as a dict {"Player": name, "Form": value}
            players_data = [{"Player": name, "Form": round(form, 1)} for form, name, pos in players]
            players_data.append({"Daily Total": round(form_sum, 1)})

            weekly_sched[day] = players_data

        return weekly_sched

    
    # def find_best_weekly_substitutions(self, extra_salary=0, top_n=5):
    #     print("🚀 Fast weekly substitution starting...")
    #     print(self.my_team.columns)
    #     print(self.best_filter.columns)
    #     extra_salary = float(extra_salary)
    #     self.my_team = self.my_team.rename(columns={"$": "Salary"})
    #     self.best_filter = self.best_filter.rename(columns={"$": "Salary"})
    #     available_players = self.best_filter[~self.best_filter["Player"].isin(self.my_team["Player"])].copy()
    #     available_players = available_players.sort_values(by="Form", ascending=False)

    #     team_matrix = self.build_player_day_matrix(copy.deepcopy(self.my_team))
    #     pool_matrix = self.build_player_day_matrix(available_players)

    #     base_lineup = self.build_day_lineup(team_matrix)
    #     current_form = sum(base_lineup[day]["form_total"] for day in range(1, 8))
    #     current_salary = self.my_team["Salary"].sum()
    #     count = 0
    #     top_swaps = []
    #     heapq.heapify(top_swaps)
    #     min_top_form = float("-inf")

    #     valid_in_combos = defaultdict(list)

    #     for player in pool_matrix.itertuples(index=False):
    #         key = "F" if player.Pos == "front" else "B"
    #         valid_in_combos[key].append((
    #             #pd.DataFrame([player._asdict()]),
    #             player._asdict(),
    #             player.Salary
    #         ))


    #     for p1, p2 in itertools.combinations(pool_matrix.itertuples(index=False), 2):
    #         pos_tuple = tuple(sorted([p1.Pos, p2.Pos]))
    #         key = (
    #             "FF" if pos_tuple == ("front", "front") else
    #             "BB" if pos_tuple == ("back", "back") else "FB"
    #         )
    #         valid_in_combos[key].append((
    #             [p1._asdict(), p2._asdict()],
    #             p1.Salary + p2.Salary
    #         ))
    #     salary_list = {}
    #     for key in valid_in_combos:
    #         valid_in_combos[key].sort(key=lambda x: x[1])
    #         salary_list[key] = [salary for (_, salary) in valid_in_combos[key]]

    #     for num_swaps in [1, 2]:
    #         for out_players in itertools.combinations(team_matrix.itertuples(index=False, name="Row"), num_swaps):
    #             #out_df = pd.DataFrame(out_players)
    #             #out_df.columns = team_matrix.columns
    #             #out_salary = out_df["Salary"].sum()
    #             out_list = [p._asdict() for p in out_players]
    #             out_salary = sum(p["Salary"] for p in out_list)


    #             if num_swaps == 1:
    #                 pos = out_list[0]["Pos"]
    #                 key = "F" if pos == "front" else "B"
    #                 #in_pool = valid_in_combos[key]
    #             else:
    #                 #pos_tuple = tuple(sorted(p["Pos"].str.lower() for p in out_list))
    #                 pos_tuple = tuple(sorted(p["Pos"].lower() for p in out_list))
    #                 key = (
    #                     "FF" if pos_tuple == ("front", "front") else
    #                     "BB" if pos_tuple == ("back", "back") else "FB"
    #                 )
    #                 #in_pool = valid_in_combos[key]
    #             #valid_swaps = [(in_players, salary) for (in_players, salary) in in_pool if salary <= (out_salary + extra_salary)]
    #             max_salary = out_salary + extra_salary
    #             idx = bisect.bisect_right(salary_list[key], max_salary)
    #             valid_swaps = valid_in_combos[key][:idx]
    #             if not valid_swaps:
    #                 continue

    #             #removed_lineup = copy.deepcopy(base_lineup)
    #             removed_lineup = self.clone_lineup(base_lineup)
    #             for row in out_list:
    #                 self.update_day_lineup_remove(removed_lineup, row)
                
    #             for in_players, salary in valid_swaps:
    #                 count+=1
    #                 in_list = in_players if isinstance(in_players, list) else [in_players]
    #                 new_form = self.evaluate_swap_fast(in_list, removed_lineup)
    #                 if new_form > min_top_form:
    #                     new_salary = current_salary - out_salary + salary
    #                     if len(top_swaps) < top_n:
    #                         heapq.heappush(top_swaps, (new_form,new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list]))
    #                     else:
    #                         heapq.heappushpop(top_swaps, (new_form, new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list]))
    #                     min_top_form = top_swaps[0][0]

    #     top_swaps.sort(reverse=True)
    #     print(f"total swaps optionable {count}")
    #     # Now attach the weekly schedule for the final top swaps
    #     final_results = []
    #     for score, salary, out_players, in_players in top_swaps:
    #         #new_team_df = self.my_team[~self.my_team["Player"].isin(out_players)].copy()
    #         #in_df = self.best_filter[self.best_filter["Player"].isin(in_players)]
    #         #final_team = pd.concat([new_team_df, in_df], ignore_index=True)
    #         new_team_dicts = [p for p in self.my_team.to_dict(orient="records") if p["Player"] not in out_players]
    #         in_dicts = [p for p in self.best_filter.to_dict(orient="records") if p["Player"] in in_players]
    #         final_team = pd.DataFrame(new_team_dicts + in_dicts)
    #         start_sched = time.perf_counter()
    #         weekly_sched = self.print_weekly_form(final_team)
    #         print(f"🕒 Weekly form took: {time.perf_counter() - start_sched:.2f} sec")
    #         final_results.append((
    #             round(score, 2),
    #             round(current_form, 2),
    #             round(salary,1),
    #             out_players,
    #             in_players,
    #             final_team.reset_index(drop=True).to_dict(orient="records"),
    #             weekly_sched
    #         ))
    #     print("subs")
    #     for swap in final_results:
    #         print("DEBUG: Swap Structure")
    #         for i, item in enumerate(swap):
    #             print(f" - swap[{i}] ({type(item)}): {item}")
    #     print("🏁 Substitution search completed.")
    #     return final_results
    
    
    
    
    
    
    
    
    
    
    
    







    
    
    
    





















    # second week functions
    def build_player_day_matrix_second(self, df=None):
        """
        Adds columns Day1–Day7 to the given DataFrame (default: full pool),
        where each column represents the player's form if they're playing that day.
        """
        if df is None:
            df = self.best_filter.copy()

        df["Form"] = df["Form"].astype(float)
        df["Pos"] = df["Pos"].str.lower()

        for day in range(1, 8):
            playing_teams = self.playing_teams_second_dict.get(day, [])
            df[f"Day{day}"] = df.apply(
                lambda row: row["Form"] if row["team"] in playing_teams else 0.0,
                axis=1
            )
        print("✅ Player-day matrix built with Day1–Day7 columns.")
        return df
    
    def print_weekly_form_second(self, team_df=None):
        if team_df is None:
            team_df = self.my_team  # Default to current team if not provided

        # Create player-day matrix (adds Day1 to Day7 columns)
        team_matrix = self.build_player_day_matrix_second(team_df)

        # Build daily best lineups (5 best players per day with max 3 front and 3 back)
        day_lineups = self.build_day_lineup(team_matrix)

        weekly_sched = {}

        for day in range(1, 8):
            players = day_lineups[day]["total"]
            form_sum = day_lineups[day]["form_total"]

            # Format each player as a dict {"Player": name, "Form": value}
            players_data = [{"Player": name, "Form": round(form, 1)} for form, name, pos in players]
            players_data.append({"Daily Total": round(form_sum, 1)})

            weekly_sched[day] = players_data

        return weekly_sched
    
    def find_best_weekly_substitutions(self, extra_salary=0, top_n=5):
        print("🚀 Fast weekly substitution starting...")
        print(self.my_team.columns)
        print(self.best_filter.columns)
        extra_salary = float(extra_salary)
        self.my_team = self.my_team.rename(columns={"$": "Salary"})
        self.best_filter = self.best_filter.rename(columns={"$": "Salary"})
        available_players = self.pool_matrix_week1[~self.pool_matrix_week1["Player"].isin(self.my_team["Player"])].copy()
        available_players = available_players.sort_values(by="Form", ascending=False)

        team_matrix = self.build_player_day_matrix(copy.deepcopy(self.my_team))
        pool_matrix = available_players

        # delete later
        # team_matrix["Day4"] = 0.0
        # pool_matrix["Day4"] = 0.0

        base_lineup = self.build_day_lineup(team_matrix)
        current_form = sum(base_lineup[day]["form_total"] for day in range(1, 8))
        #current_form = sum(base_lineup[day]["form_total"] for day in list(range(1, 4)) + list(range(5, 8)))
        current_salary = self.my_team["Salary"].sum()
        count = 0
        top_swaps = []
        heapq.heapify(top_swaps)
        
        heapq.heappush(top_swaps, (current_form, current_salary, [], []))
        min_top_form = current_form

        valid_in_combos = defaultdict(list)

        for player in pool_matrix.itertuples(index=False):
            key = "F" if player.Pos == "front" else "B"
            valid_in_combos[key].append((
                #pd.DataFrame([player._asdict()]),
                player._asdict(),
                player.Salary
            ))


        for p1, p2 in itertools.combinations(pool_matrix.itertuples(index=False), 2):
            pos_tuple = tuple(sorted([p1.Pos, p2.Pos]))
            key = (
                "FF" if pos_tuple == ("front", "front") else
                "BB" if pos_tuple == ("back", "back") else "FB"
            )
            valid_in_combos[key].append((
                [p1._asdict(), p2._asdict()],
                p1.Salary + p2.Salary
            ))
        salary_list = {}
        for key in valid_in_combos:
            valid_in_combos[key].sort(key=lambda x: x[1])
            salary_list[key] = [salary for (_, salary) in valid_in_combos[key]]

        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(team_matrix.itertuples(index=False, name="Row"), num_swaps):
                #out_df = pd.DataFrame(out_players)
                #out_df.columns = team_matrix.columns
                #out_salary = out_df["Salary"].sum()
                out_list = [p._asdict() for p in out_players]
                out_salary = sum(p["Salary"] for p in out_list)


                if num_swaps == 1:
                    pos = out_list[0]["Pos"]
                    key = "F" if pos == "front" else "B"
                    #in_pool = valid_in_combos[key]
                else:
                    #pos_tuple = tuple(sorted(p["Pos"].str.lower() for p in out_list))
                    pos_tuple = tuple(sorted(p["Pos"].lower() for p in out_list))
                    key = (
                        "FF" if pos_tuple == ("front", "front") else
                        "BB" if pos_tuple == ("back", "back") else "FB"
                    )
                    #in_pool = valid_in_combos[key]
                #valid_swaps = [(in_players, salary) for (in_players, salary) in in_pool if salary <= (out_salary + extra_salary)]
                max_salary = out_salary + extra_salary
                idx = bisect.bisect_right(salary_list[key], max_salary)
                valid_swaps = valid_in_combos[key][:idx]
                if not valid_swaps:
                    continue

                #removed_lineup = copy.deepcopy(base_lineup)
                removed_lineup = self.clone_lineup(base_lineup)
                for row in out_list:
                    self.update_day_lineup_remove(removed_lineup, row)
                
                for in_players, salary in valid_swaps:
                    count+=1
                    in_list = in_players if isinstance(in_players, list) else [in_players]
                    new_form = self.evaluate_swap_fast(in_list, removed_lineup)
                    if new_form > min_top_form:
                        new_salary = current_salary - out_salary + salary
                        if len(top_swaps) < top_n:
                            heapq.heappush(top_swaps, (new_form,new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list]))
                        else:
                            heapq.heappushpop(top_swaps, (new_form, new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list]))
                        min_top_form = top_swaps[0][0]

        top_swaps.sort(reverse=True)
        print(f"total swaps optionable {count}")
        # Now attach the weekly schedule for the final top swaps
        final_results = []
        for score, salary, out_players, in_players in top_swaps:
            #new_team_df = self.my_team[~self.my_team["Player"].isin(out_players)].copy()
            #in_df = self.best_filter[self.best_filter["Player"].isin(in_players)]
            #final_team = pd.concat([new_team_df, in_df], ignore_index=True)
            new_team_dicts = [p for p in self.my_team.to_dict(orient="records") if p["Player"] not in out_players]
            in_dicts = [p for p in self.best_filter.to_dict(orient="records") if p["Player"] in in_players]
            final_team = pd.DataFrame(new_team_dicts + in_dicts)
            start_sched = time.perf_counter()
            weekly_sched = self.print_weekly_form(final_team)
            print(f"🕒 Weekly form took: {time.perf_counter() - start_sched:.2f} sec")
            final_results.append((
                round(score, 2),
                round(current_form, 2),
                round(salary,1),
                out_players,
                in_players,
                final_team.reset_index(drop=True).to_dict(orient="records"),
                weekly_sched
            ))
        print("subs")
        for swap in final_results:
            print("DEBUG: Swap Structure")
            for i, item in enumerate(swap):
                print(f" - swap[{i}] ({type(item)}): {item}")
        print("🏁 Substitution search completed.")
        return final_results
    

    def find_best_weekly_substitutions_second(self, extra_salary=0, top_n=1, current_team_df=None, max_total_salary=None):
        """
        Weekly substitution search using the SECOND week schedule (schedule_second).
        Optionally:
          - current_team_df: use a custom starting team instead of self.my_team
          - max_total_salary: global salary cap; will reduce effective extra_salary
        Returns: list of tuples with same structure as find_best_weekly_substitutions.
        """
        if self.schedule_second is None:
            raise ValueError("Second-week schedule not provided to FantasyOptimizer.")

        print("🚀 Second-week fast weekly substitution starting...")

        extra_salary = float(extra_salary)
        # allow passing a different starting team
        if current_team_df is not None:
            self.my_team = current_team_df.copy()

        self.my_team = self.my_team.rename(columns={"$": "Salary"})
        self.best_filter = self.best_filter.rename(columns={"$": "Salary"})

        self.my_team["Salary"] = self.my_team["Salary"].astype(float)
        self.best_filter["Salary"] = self.best_filter["Salary"].astype(float)

        if self.pool_matrix_week2 is None:
            self.pool_matrix_week2 = self.build_player_day_matrix_second(self.best_filter.copy())

        # remove current team from pool
        available_players = self.pool_matrix_week2[~self.pool_matrix_week2["Player"].isin(current_team_df["Player"])].copy()
        available_players = available_players.sort_values(by="Form", ascending=False)

        team_matrix = self.build_player_day_matrix_second(current_team_df.copy())
        pool_matrix = available_players

        base_lineup = self.build_day_lineup(team_matrix)
        current_form = sum(base_lineup[day]["form_total"] for day in range(1, 8))
        current_salary = self.my_team["Salary"].sum()

        # If we have a global cap, adjust effective extra_salary
        if max_total_salary is not None:
            extra_salary = max(0.0, float(max_total_salary) - float(current_salary))

        count = 0
        top_swaps = []
        heapq.heapify(top_swaps)

        # no subs option
        heapq.heappush(top_swaps, (current_form, current_salary, [], []))
        min_top_form = current_form

        valid_in_combos = defaultdict(list)

        for player in pool_matrix.itertuples(index=False):
            key = "F" if player.Pos == "front" else "B"
            valid_in_combos[key].append((player._asdict(), player.Salary))

        for p1, p2 in itertools.combinations(pool_matrix.itertuples(index=False), 2):
            pos_tuple = tuple(sorted([p1.Pos, p2.Pos]))
            key = (
                "FF" if pos_tuple == ("front", "front") else
                "BB" if pos_tuple == ("back", "back") else "FB"
            )
            valid_in_combos[key].append(([p1._asdict(), p2._asdict()], p1.Salary + p2.Salary))

        salary_list = {}
        for key in valid_in_combos:
            valid_in_combos[key].sort(key=lambda x: x[1])
            salary_list[key] = [salary for (_, salary) in valid_in_combos[key]]

        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(team_matrix.itertuples(index=False, name="Row"), num_swaps):
                out_list = [p._asdict() for p in out_players]
                out_salary = sum(p["Salary"] for p in out_list)

                if num_swaps == 1:
                    pos = out_list[0]["Pos"]
                    key = "F" if pos == "front" else "B"
                else:
                    pos_tuple = tuple(sorted(p["Pos"].lower() for p in out_list))
                    key = (
                        "FF" if pos_tuple == ("front", "front") else
                        "BB" if pos_tuple == ("back", "back") else "FB"
                    )

                max_salary = out_salary + extra_salary
                idx = bisect.bisect_right(salary_list[key], max_salary)
                valid_swaps = valid_in_combos[key][:idx]
                if not valid_swaps:
                    continue

                removed_lineup = self.clone_lineup(base_lineup)
                for row in out_list:
                    self.update_day_lineup_remove(removed_lineup, row)

                for in_players, salary in valid_swaps:
                    count += 1
                    in_list = in_players if isinstance(in_players, list) else [in_players]
                    new_form = self.evaluate_swap_fast(in_list, removed_lineup)
                    if new_form > min_top_form:
                        new_salary = current_salary - out_salary + salary
                        if max_total_salary is not None and new_salary > max_total_salary:
                            continue

                        if len(top_swaps) < top_n:
                            heapq.heappush(
                                top_swaps,
                                (new_form, new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list])
                            )
                        else:
                            heapq.heappushpop(
                                top_swaps,
                                (new_form, new_salary, [p["Player"] for p in out_list], [p["Player"] for p in in_list])
                            )
                        min_top_form = top_swaps[0][0]

        top_swaps.sort(reverse=True)
        print(f"total second-week swaps optionable {count}")

        final_results = []
        for score, salary, out_players, in_players in top_swaps:
            new_team_dicts = [p for p in self.my_team.to_dict(orient="records") if p["Player"] not in out_players]
            in_dicts = [p for p in self.best_filter.to_dict(orient="records") if p["Player"] in in_players]
            final_team = pd.DataFrame(new_team_dicts + in_dicts)
            start_sched = time.perf_counter()
            weekly_sched = self.print_weekly_form_second(final_team)
            print(f"🕒 Second-week weekly form took: {time.perf_counter() - start_sched:.2f} sec")
            final_results.append((
                round(score, 2),
                round(current_form, 2),
                round(salary, 1),
                out_players,
                in_players,
                final_team.reset_index(drop=True).to_dict(orient="records"),
                weekly_sched
            ))

        print("🏁 Second-week substitution search completed.")
        return final_results

    

    def find_best_Biweekly_substitutions(self, extra_salary=0, top_n=5):
        """
        Find best bi-weekly plan:
            - 0–2 subs before Week 1 (using first schedule)
            - 0–2 subs before Week 2 (using second schedule)
            - Week 2 starts from the Week 1 result team
            - Total score = week1_new_form + week2_new_form
            - Global salary cap: final salary <= start_salary + extra_salary
        """
        if self.schedule_second is None:
            raise ValueError("Second-week schedule not provided to FantasyOptimizer.")

        print("🚀 Bi-weekly substitution search starting...")

        counter = itertools.count()
        extra_salary = float(extra_salary)
        self.my_team = self.my_team.rename(columns={"$": "Salary"})
        self.best_filter = self.best_filter.rename(columns={"$": "Salary"})

        self.my_team["Salary"] = self.my_team["Salary"].astype(float)
        self.best_filter["Salary"] = self.best_filter["Salary"].astype(float)

        start_salary = self.my_team["Salary"].sum()
        max_total_salary = start_salary + extra_salary

        # --- 1. Week 1 candidates (use weekly function with a slightly larger N) ---
        week1_candidates_n = max(top_n * 4, top_n + 10)
        week1_swaps = self.find_best_weekly_substitutions(extra_salary=extra_salary, top_n=week1_candidates_n)

        if not week1_swaps:
            print("⚠️ No week-1 swaps found, falling back to no-sub plan only.")
            # build baseline
            base_sched = self.print_weekly_form(self.my_team)
            base_form = sum(d[-1]["Daily Total"] for d in base_sched.values())
            week1_swaps = [(base_form, base_form, start_salary, [], [], self.my_team.to_dict(orient="records"), base_sched)]

        biweekly_heap = []
        heapq.heapify(biweekly_heap)

        # current_form (week-1 baseline) is the same for all week1 swaps
        week1_current_form = week1_swaps[0][1]
        min_top_form = float("-inf")
        for idx, swap1 in enumerate(week1_swaps):
            week1_new_form, _, week1_salary, out1, in1, team1_records, week1_sched = swap1
            team1_df = pd.DataFrame(team1_records)

            # --- 2. Week 2 optimization from team1_df ---
            remaining_extra_for_week2 = max(0.0, max_total_salary - week1_salary)

            optimizer_week2 = FantasyOptimizer(team1_df, self.best_filter, self.schedule, self.schedule_second)
            week2_swaps = optimizer_week2.find_best_weekly_substitutions_second(
                extra_salary=remaining_extra_for_week2,
                top_n=5,
                current_team_df=team1_df,
                max_total_salary=max_total_salary
            )

            if week2_swaps:
                for n in range(0,4):
                    week2_best = week2_swaps[n]
                    week2_new_form, week2_current_form, week2_salary, out2, in2, team2_records, week2_sched = week2_best
                    total_biweekly_form = week1_new_form + week2_new_form
                    if total_biweekly_form > min_top_form:
                        plan = {
                            "total_biweekly_form": round(total_biweekly_form, 2),
                            "week1": {
                                "new_form": round(week1_new_form, 2),
                                "current_form": round(week1_current_form, 2),
                                "salary": round(week1_salary, 1),
                                "out": out1,
                                "in": in1,
                                "team": team1_records,
                                "weekly_sched": week1_sched,
                            },
                            "week2": {
                                "new_form": round(week2_new_form, 2),
                                "current_form": round(week2_current_form, 2),
                                "salary": round(week2_salary, 1),
                                "out": out2,
                                "in": in2,
                                "team": team2_records,
                                "weekly_sched": week2_sched,
                            },
                        }
                        if len(biweekly_heap) < top_n:
                            heapq.heappush(biweekly_heap, (total_biweekly_form, next(counter), plan))
                        else:
                            heapq.heappushpop(biweekly_heap, (total_biweekly_form, next(counter), plan))
                            min_top_form = biweekly_heap[0][0]
            else:
                # no subs week2 – baseline for week2
                continue

        # sort best plans descending
        biweekly_heap.sort(key=lambda x: x[0], reverse=True)
        best_plans = [plan for _, _, plan in biweekly_heap]

        print("🏁 Bi-weekly substitution search completed.")
        return best_plans

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
        batch_size = 10

        for num_swaps in [1, 2]:
            for i in range(0, len(available_players)):
                for in_players in itertools.combinations(available_players.itertuples(index=False, name="PlayerTuple"), num_swaps):
                    in_df = pd.DataFrame(in_players, columns=[["Unnamed: 0", "Player", "Salary", "Form", "TP.", "Pos", "team"]])
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

                    available_combos[key].append((float(form_sum.iloc[0]), float(salary_sum.iloc[0]), in_df))

        # **Step 2: Sort each category by salary (ascending)**
        for key in available_combos:
            available_combos[key].sort(key=lambda x: x[1])  # Sort by salary sum (low to high)
        print("# available combos computed")
        # **Step 3: Track the top N best swaps using a Min Heap**
        top_swaps = []  # Will store (form_gain, new_team, new_form, new_salary, out_list, in_list)
        min_top_form = float('-inf')  # Track the smallest form gain in the heap

        # **Step 4: Iterate through all possible user team swap combinations (1 and 2 swaps)**
        for num_swaps in [1, 2]:
            for out_players in itertools.combinations(self.my_team.itertuples(index=False, name="PlayerTuple"), num_swaps):
                out_list = list(out_players)
                available_salary = sum(float(player.Salary) for player in out_list) + extra_salary
                form_out = sum(float(player.Form) for player in out_list)

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
                valid_swaps = [(form, salary, df) for (form, salary, df) in swap_list if float(salary) <= available_salary]
                if not valid_swaps:
                    continue  # Skip if no valid swaps exist

                # **Step 6: Pick the best swap (max form sum)**
                best_form_swap = max(valid_swaps, key=lambda x: x[0])  # Get max form sum
                form_sum, salary_sum, in_df = best_form_swap

                # **Step 7: Compute form gain EARLY (before creating new team)**
                form_gain = form_sum - form_out


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
                        round(float(current_form), 1),  
                        round(float(new_salary), 1),  
                        [p.Player for p in out_list], 
                        in_df["Player"].iloc[:, 0].tolist(),
                        new_team.to_dict(orient="records")  # ✅ Move to the end
                    ))
                else:
                    heapq.heappushpop(top_swaps, (
                        round(float(form_gain), 1),  
                        round(float(new_form), 1),
                        round(float(current_form), 1),  
                        round(float(new_salary), 1),  
                        [p.Player for p in out_list], 
                        in_df["Player"].iloc[:, 0].tolist(),
                        new_team.to_dict(orient="records")  # ✅ Move to the end
                    ))
                # **Update min_top_form**
                min_top_form = top_swaps[0][0]  # The smallest form gain in the heap


        # **Step 10: Get the top N best swaps (sorted by highest form gain)**
        top_swaps.sort(reverse=True, key=lambda x: x[0])  # Sort by highest form gain

        return top_swaps  # Returns a list of top swaps instead of just the best one




