from flask import Flask, render_template, request, jsonify, session
from flask_session import Session  # Import Flask-Session
import pandas as pd
import numpy as np
import json
from optimizer import FantasyOptimizer
from optimizer_prefix_suffix import FantasyOptimizerPrefixSuffix
from best_team_finder import BestTeamFinder
from wildcard import WildcardOptimizer
from data_loader import data_loader
import time
import gc

app = Flask(__name__, template_folder="templates", static_folder="templates/static")
app.secret_key = "super_secret_key"

# Store session on the server instead of cookies
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./.flask_session/"  # Store session files locally
app.config["SECRET_KEY"] = "your_secret_key_here"  # Required for session security
Session(app)  # Initialize server-side session
data_load = data_loader()
data_load.load_teams()
schedule_payload = data_load.get_or_update_schedule(num_weeks=4)
app.config["SCHEDULE_DATA"] = schedule_payload["weeks"]
#data_load.load_schedule()

## changing temporarily for showing my code
FULL_PLAYERS_CSV = "data/top_update_players_2026.csv"
## FULL_PLAYERS_CSV = "data/top_update_players(1).csv"
#SCHEDULE_CSV = "data/GWSchedule26_11.csv"
#SECOND_SCHEDULE_CSV ="data/GWSchedule26_11.csv"
BIWEEKLY_SCHEDULE_CSV = "data/GWSchedule26_4.csv"

full_players_df = pd.read_csv(FULL_PLAYERS_CSV)

# Update Deni Avdija's form to 40 (temporary override)
if "Jalen Duren" in full_players_df["Player"].values:
    full_players_df.loc[full_players_df["Player"] == "Jalen Duren", "Form"] = 40
    print("✅ Updated Jalen Duren form to 40 in full_players_df")

# Add Deni Avdija manually (not in dataset)
# deni_avdija_row = pd.DataFrame([{
#     "Unnamed: 0": 285,
#     "Player": "Deni Avdija",
#     "$": 12.9,
#     "Form": 14.6,
#     "TP.": 2211,
#     "Pos": "front",
#     "team": "POR"
# }])
# full_players_df = pd.concat([full_players_df, deni_avdija_row], ignore_index=True)

def get_best_team(salary_cap=100):
    best_team_finder = BestTeamFinder(full_players_df, salary_cap)
    return best_team_finder.find_best_team()

def replace_nan_with_none(obj):
    """Recursively replace NaN values with None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None  # JSON does not support NaN, so we replace it with None
    return obj

def get_schedule_paths():
    current_gw = session.get("current_gw")
    next_gw = session.get("next_gw")

    if current_gw is None or next_gw is None:
        raise ValueError("Week selection not set")

    sched_1 = f"data/GWSchedule_{current_gw}.csv"
    sched_2 = f"data/GWSchedule_{next_gw}.csv"

    return sched_1, sched_2

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("query", "").lower()
    if not query:
        return jsonify([])

    suggestions = full_players_df[full_players_df["Player"].str.lower().str.contains(query, na=False)]
    return jsonify(suggestions["Player"].tolist())

@app.before_request
def reset_session_if_needed():
    """Clears session-stored data if no user team exists."""
    if "user_team" not in session:  # No user team? Reset stored players
        session.pop("updated_players_df", None)


@app.route("/set_user_team", methods=["POST"])
def set_user_team():
    data = request.get_json()
    user_team = data.get("user_team", [])
    extra_salary = float(data.get("extra_salary", 0))
    updated_players_df = full_players_df.copy()

    # for player in user_team:
    #     updated_players_df.loc[updated_players_df["Player"] == player["Player"], "$"] = player["$"]

    # Match players from session data
    for player in user_team:
        matching_row = updated_players_df.loc[updated_players_df["Player"] == player.get("Player")]

        if matching_row.empty:
            print(f"⚠️ Player {player.get('Player')} not found in updated dataset.")
        else:
            # Extract the first row from the filtered DataFrame
            row_data = matching_row.iloc[0]

            player["Unnamed: 0"] = row_data.get("Unnamed: 0", "unknown")
            player["Pos"] = row_data.get("Pos", "unknown")
            player["Form"] = row_data.get("Form", "unknown")
            # Keep user's custom salary as '$' (what they paid / value player at)
            # The optimizer will look up the original market price from best_filter
            player['$'] = player['$']
            player["TP."] = row_data.get("TP.", "unknown")
            player["team"] = row_data.get("team", "unknown")

    
    # Save user team in session
    session["updated_players_df"] = updated_players_df
    session["user_team"] = user_team
    session["extra_salary"] = extra_salary
    session.modified=True

    return jsonify({"message": "✅ User team saved successfully!"})

def clean_dict_keys(data):
    """Remove tuple keys from a dictionary."""
    if isinstance(data, list):
        return [clean_dict_keys(item) for item in data]
    elif isinstance(data, dict):
        return {str(key) if isinstance(key, tuple) else key: clean_dict_keys(value) for key, value in data.items()}
    return data

def convert_floats(obj):
    """Recursively convert numpy float types to regular floats."""
    if isinstance(obj, dict):
        return {key: convert_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(value) for value in obj]
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)  # ✅ Convert NumPy float to Python float
    return obj

@app.route("/set_week", methods=["POST"])
def set_week():
    data = request.get_json()

    gw1 = str(data["current_week"])
    gw2 = str(data["next_week"])

    schedules = app.config["SCHEDULE_DATA"]

    if gw1 not in schedules or gw2 not in schedules:
        return jsonify({"error": f"Selected week not available. Available weeks: {list(schedules.keys())}"}), 400

    session["schedule_week1"] = schedules[gw1]
    session["schedule_week2"] = schedules[gw2]
    session["selected_gw1"] = gw1  # Store the week number too for display
    session["selected_gw2"] = gw2
    session.modified = True  # Ensure Flask-Session saves the changes

    print(f"✅ Set week selection: GW{gw1} and GW{gw2}")
    return jsonify({"message": f"✅ Week selection saved: GW{gw1} and GW{gw2}"})


@app.route("/get_available_weeks", methods=["GET"])
def get_available_weeks():
    """Return available gameweeks from schedule data."""
    schedules = app.config["SCHEDULE_DATA"]
    available = sorted(schedules.keys(), key=lambda x: int(x))
    return jsonify({"weeks": available})


@app.route("/compute_result", methods=["POST"])
def compute_result():
    gc.collect()
    data = request.json
    option = data.get("option")
    sub_type = data.get("sub_type", "weekly")
    salary_cap = float(data.get("salary_cap", 100))
    top_n = int(data.get("top_n", 5))

    print(f"📢 Compute Request: {data}")
    user_team_data = session.get("user_team")
    extra_salary = float(session.get("extra_salary", 0))
    if "user_team" in session and session.get("updated_players_df") is not None:
        updated_players_data = pd.DataFrame(session["updated_players_df"])
    else:
        updated_players_data = full_players_df.copy()  # Use fresh dataset if no user team exists
    if not "user_team" in session:
        print("user team not in session")
    # **Fix missing/empty DataFrame cases**
    if updated_players_data.empty:
        print("❌ ERROR: No valid dataset found. Using original dataset.")
        updated_players_data = full_players_df.copy()

    response_data = {}

    user_team = pd.DataFrame(user_team_data)
    user_team = user_team[["Player", "$", "Form", "TP.", "Pos", "team"]]
    updated_players_df = pd.DataFrame(updated_players_data)
    updated_players_df = updated_players_df[["Player", "$", "Form", "TP.", "Pos","team"]]

    if option == "best_team":
        # ✅ Allow running best team even if no team was entered
        best_team_finder = BestTeamFinder(pd.DataFrame(updated_players_data), salary_cap)
        best_team = best_team_finder.find_best_team()

        if best_team is None or best_team.empty:
            return jsonify({"error": "⚠️ No valid team found."})

        response_data = {
            "best_team": best_team.to_dict(orient="records"),
            "total_form": best_team["Form"].sum().round(1),
            "total_price": best_team["$"].sum().round(1)
        }
        return jsonify(response_data)
    
    # NEW: Added handler for bi-weekly team finder
    elif option == "best_biweekly_team":
        try:
            biweekly_schedule_df = pd.read_csv(BIWEEKLY_SCHEDULE_CSV)
        except FileNotFoundError:
            return jsonify({"error": f"⚠️ Bi-weekly schedule file not found at {BIWEEKLY_SCHEDULE_CSV}"})

        best_team_finder = BestTeamFinder(full_players_df.copy(), salary_cap)
        top_teams = best_team_finder.find_best_biweekly_team(biweekly_schedule_df, top_n=5)

        if not top_teams:
            return jsonify({"error": "⚠️ Could not generate any bi-weekly teams."})

        # Convert teams from DataFrames to dictionaries for JSON response
        response_data["top_teams"] = []
        for item in top_teams:
            team_df = item['team']
            response_data["top_teams"].append({
                "score": round(item['score'], 2),
                "team": team_df.to_dict(orient="records"),
                "total_price": team_df["$"].sum().round(1)
            })
        return jsonify(response_data)

    elif option == "best_substitutions":
        if not user_team_data:
            return jsonify({"error": "⚠️ Please enter your team first!"})
        start_time = time.time()
        # optimizer = FantasyOptimizer(pd.DataFrame(user_team_data), pd.DataFrame(updated_players_data), SCHEDULE_CSV, SECOND_SCHEDULE_CSV)
        # best_swaps = optimizer.find_best_weekly_substitutions(extra_salary, top_n) if sub_type == "weekly" else optimizer.find_best_total_substitutions(extra_salary, top_n)

        # Extract new parameters from request
        untradable_players = data.get("untradable_players", [])
        must_trade_players = data.get("must_trade_players", [])
        injured_pool_players = data.get("injured_pool_players", [])
        sub_day = data.get("sub_day")
        if sub_day:
            sub_day = int(sub_day)  # Convert to int (1-7)
        
        # Filter out injured pool players from the dataset
        if injured_pool_players:
            updated_players_data = updated_players_data[~updated_players_data["Player"].isin(injured_pool_players)]
            print(f"🏥 Excluded {len(injured_pool_players)} injured players from pool: {injured_pool_players}")
        
        schedule_week1 = session.get("schedule_week1")
        schedule_week2 = session.get("schedule_week2")
        
        # Fallback to first two weeks from SCHEDULE_DATA if session schedules are not set
        if schedule_week1 is None or schedule_week2 is None:
            schedules = app.config["SCHEDULE_DATA"]
            available_weeks = sorted(schedules.keys(), key=lambda x: int(x))
            if len(available_weeks) >= 2:
                if schedule_week1 is None:
                    schedule_week1 = schedules[available_weeks[0]]
                    print(f"⚠️ Using default schedule_week1: week {available_weeks[0]}")
                if schedule_week2 is None:
                    schedule_week2 = schedules[available_weeks[1]]
                    print(f"⚠️ Using default schedule_week2: week {available_weeks[1]}")
            else:
                return jsonify({"error": "⚠️ No schedule data available. Please refresh the page."})
        
        # Convert string keys to int keys if needed
        if schedule_week1 and isinstance(list(schedule_week1.keys())[0] if schedule_week1.keys() else None, str):
            schedule_week1 = {int(k): v for k, v in schedule_week1.items()}
        if schedule_week2 and isinstance(list(schedule_week2.keys())[0] if schedule_week2.keys() else None, str):
            schedule_week2 = {int(k): v for k, v in schedule_week2.items()}

        optimizer = FantasyOptimizer(
            pd.DataFrame(user_team_data),
            pd.DataFrame(updated_players_data),
            schedule_week1,
            schedule_week2
        )


        if sub_type == "weekly":
            best_swaps = optimizer.find_best_weekly_substitutions(
                extra_salary, top_n,
                untradable_players=untradable_players,
                must_trade_players=must_trade_players,
                sub_day=sub_day
            )
        elif sub_type == "weekly_anyday":
            # Use new prefix-suffix optimizer with any-day substitutions
            optimizer_ps = FantasyOptimizerPrefixSuffix(
                pd.DataFrame(user_team_data),
                pd.DataFrame(updated_players_data),
                schedule_week1,
                schedule_week2
            )
            # Get week weights from request (default 75/25)
            w1 = float(data.get("week1_weight", 0.75))
            w2 = float(data.get("week2_weight", 0.25))
            
            results_dict = optimizer_ps.find_best_weekly_substitutions_any_day(
                extra_salary=extra_salary,
                top_n=top_n,
                untradable_players=untradable_players,
                must_trade_players=must_trade_players,
                w1=w1,
                w2=w2,
                allow_domination_prune=True,
                shortlist_k=60
            )
            # Results are already in dict format, convert for response
            best_swaps = results_dict  # Keep as dict for new display format
        elif sub_type == "late_week_sub":
            # Late week substitution: user is on day X of current week with 1-2 subs remaining
            # Score = 0.8 * (partial current week + next full week) + 0.2 * week after
            
            late_week_day = data.get("late_week_day")
            max_subs = data.get("subs_remaining", 2)
            if late_week_day:
                late_week_day = int(late_week_day)
            else:
                late_week_day = 7  # Default to day 7
            max_subs = int(max_subs)
            
            # Get 3 weeks of schedule
            schedules = app.config["SCHEDULE_DATA"]
            available_weeks = sorted(schedules.keys(), key=lambda x: int(x))
            
            # Get user-selected weeks or derive from session
            selected_gw1 = session.get("selected_gw1")
            selected_gw2 = session.get("selected_gw2")
            
            if selected_gw1 is None or selected_gw2 is None:
                if len(available_weeks) < 3:
                    return jsonify({"error": "⚠️ Need at least 3 weeks of schedule data for late week sub."})
                partial_week_key = available_weeks[0]
                next_week_key = available_weeks[1]
                week_after_key = available_weeks[2]
            else:
                # selected_gw1 is current partial week, selected_gw2 is next week
                partial_week_key = str(selected_gw1)
                next_week_key = str(selected_gw2)
                
                # Find week after selected_gw2
                try:
                    idx = available_weeks.index(next_week_key)
                    if idx + 1 < len(available_weeks):
                        week_after_key = available_weeks[idx + 1]
                    else:
                        week_after_key = next_week_key  # Fallback to same week if no more weeks
                except ValueError:
                    return jsonify({"error": f"⚠️ Selected week {next_week_key} not found in schedule."})
            
            partial_week_schedule = schedules.get(partial_week_key, {})
            next_week_schedule = schedules.get(next_week_key, {})
            week_after_schedule = schedules.get(week_after_key, {})
            
            print(f"📅 Late week sub: partial week {partial_week_key} (day {late_week_day}), next week {next_week_key}, week after {week_after_key}")
            
            # Use prefix-suffix optimizer for late week subs
            optimizer_ps = FantasyOptimizerPrefixSuffix(
                pd.DataFrame(user_team_data),
                pd.DataFrame(updated_players_data),
                next_week_schedule,
                week_after_schedule
            )
            
            # Get weights
            w1 = float(data.get("week1_weight", 0.8))
            w2 = float(data.get("week2_weight", 0.2))
            
            results_dict = optimizer_ps.find_best_late_week_substitutions(
                partial_week_schedule=partial_week_schedule,
                next_week_schedule=next_week_schedule,
                week_after_schedule=week_after_schedule,
                late_week_day=late_week_day,
                max_subs=max_subs,
                extra_salary=extra_salary,
                top_n=top_n,
                untradable_players=untradable_players,
                must_trade_players=must_trade_players,
                w1=w1,
                w2=w2,
                allow_domination_prune=True,
                shortlist_k=60
            )
            best_swaps = results_dict
        elif sub_type == "biweekly":
            best_swaps = optimizer.find_best_Biweekly_substitutions(
                extra_salary, top_n,
                untradable_players=untradable_players,
                must_trade_players=must_trade_players,
                sub_day=sub_day
            )
        else:
            return jsonify({"error": "⚠️ Invalid substitution type."})
        
        end_time = time.time()
        print(f"optimizer runtime: {end_time - start_time:.2f} seconds")
        gc.collect()

        if not best_swaps:
            return jsonify({"error": "⚠️ No valid substitutions found."})
        # if sub_type == "weekly":

        #     response_data["top_substitutions"] = [
        #         {
        #             "new_form" : swap[0],
        #             "current_form": swap[1],
        #             "new_salary": swap[2],
        #             "substitutions_out": list(swap[3]),
        #             "substitutions_in": list(swap[4]),
        #             "new_team": clean_dict_keys(swap[5]),
        #             "weekly_sched": json.dumps(convert_floats(swap[6]))
        #         }
        #         for swap in best_swaps
        #     ]
        # else:
        #     response_data["top_substitutions"] = [
        #         {
        #             "form_gain" : swap[0],
        #             "new_form" : swap[1],
        #             "current_form": swap[2],
        #             "new_salary": swap[3],
        #             "substitutions_out": swap[4],
        #             "substitutions_in": swap[5],
        #             "new_team": clean_dict_keys(swap[6]),
        #         }
        #         for swap in best_swaps
        #     ]
        if sub_type == "weekly":
            response_data["top_substitutions"] = [
                {
                    "new_form": swap[0],
                    "current_form": swap[1],
                    "new_salary": swap[2],
                    "substitutions_out": list(swap[3]),
                    "substitutions_in": list(swap[4]),
                    "new_team": clean_dict_keys(swap[5]),
                    "weekly_sched": json.dumps(convert_floats(swap[6]))
                }
                for swap in best_swaps
            ]
        elif sub_type == "weekly_anyday":
            # New format from prefix-suffix optimizer
            response_data["top_substitutions"] = []
            for result in best_swaps:
                # Extract out and in player names from subs
                out_players = [sub['out'] for sub in result['substitution_plan']['subs']]
                in_players = [sub['in'] for sub in result['substitution_plan']['subs']]
                final_team_salary = sum(p['Salary'] for p in result['final_team'])
                
                response_data["top_substitutions"].append({
                    "total_score": result['total_score'],
                    "week1_score": result['week1_score'],
                    "week2_score": result['week2_score'],
                    "num_subs": result['num_subs'],
                    "substitution_plan": result['substitution_plan'],
                    "substitutions_out": out_players,
                    "substitutions_in": in_players,
                    "new_salary": round(final_team_salary, 1),
                    "new_team": clean_dict_keys(result['final_team']),
                    "weekly_sched": json.dumps(convert_floats(result['weekly_schedule'])),
                    "legal": result['legal']
                })
        elif sub_type == "late_week_sub":
            # Late week sub format - similar to weekly_anyday but with partial week info
            response_data["top_substitutions"] = []
            for result in best_swaps:
                out_players = [sub['out'] for sub in result['substitution_plan']['subs']]
                in_players = [sub['in'] for sub in result['substitution_plan']['subs']]
                final_team_salary = sum(p['Salary'] for p in result['final_team'])
                
                response_data["top_substitutions"].append({
                    "total_score": result['total_score'],
                    "week1_score": result['week1_score'],
                    "week2_score": result['week2_score'],
                    "partial_week_score": result['partial_week_score'],
                    "next_week_score": result['next_week_score'],
                    "num_subs": result['num_subs'],
                    "substitution_plan": result['substitution_plan'],
                    "substitutions_out": out_players,
                    "substitutions_in": in_players,
                    "new_salary": round(final_team_salary, 1),
                    "new_team": clean_dict_keys(result['final_team']),
                    "weekly_sched": json.dumps(convert_floats(result['weekly_schedule'])),
                    "legal": result['legal']
                })
        elif sub_type == "biweekly":
            response_data["top_substitutions"] = []
            for plan in best_swaps:
                w1 = plan["week1"]
                w2 = plan["week2"]
                response_data["top_substitutions"].append({
                    "total_biweekly_form": plan["total_biweekly_form"],
                    "week1": {
                        "new_form": w1["new_form"],
                        "current_form": w1["current_form"],
                        "salary": w1["salary"],
                        "substitutions_out": w1["out"],
                        "substitutions_in": w1["in"],
                        "new_team": clean_dict_keys(w1["team"]),
                        "weekly_sched": json.dumps(convert_floats(w1["weekly_sched"]))
                    },
                    "week2": {
                        "new_form": w2["new_form"],
                        "current_form": w2["current_form"],
                        "salary": w2["salary"],
                        "substitutions_out": w2["out"],
                        "substitutions_in": w2["in"],
                        "new_team": clean_dict_keys(w2["team"]),
                        "weekly_sched": json.dumps(convert_floats(w2["weekly_sched"]))
                    }
                })

        
        return jsonify(response_data)

    return jsonify({"error": "⚠️ Invalid option selected."})

@app.route("/compute_wildcard", methods=["POST"])
def compute_wildcard():
    """Wildcard optimizer endpoint for finding best complete teams."""
    gc.collect()
    data = request.json
    
    # Get parameters
    budget = float(data.get("budget", 100))
    top_n = int(data.get("top_n", 5))
    must_include = data.get("must_include", [])
    must_exclude = data.get("must_exclude", [])
    injured = data.get("injured", [])
    
    # Get weights (default: 0.5, 0.3, 0.2)
    w1 = float(data.get("w1", 0.5))
    w2 = float(data.get("w2", 0.3))
    w3 = float(data.get("w3", 0.2))
    
    # Get user-selected weeks for wildcard calculation
    selected_weeks = data.get("selected_weeks", [])
    
    print(f"📢 Wildcard Request: budget={budget}, top_n={top_n}, w1={w1}, w2={w2}, w3={w3}")
    print(f"   Selected weeks: {selected_weeks}")
    print(f"   Must include: {must_include}")
    print(f"   Must exclude: {must_exclude}")
    print(f"   Injured: {injured}")
    
    # Get user team data for custom prices
    user_team_data = session.get("user_team")
    user_team_df = pd.DataFrame(user_team_data) if user_team_data else None
    
    # Get schedules
    schedules = app.config["SCHEDULE_DATA"]
    available_weeks = sorted(schedules.keys(), key=lambda x: int(x))
    
    if len(available_weeks) < 3:
        return jsonify({"error": f"⚠️ Need at least 3 weeks of schedule data. Only {len(available_weeks)} available."})
    
    # Use user-selected weeks if provided, otherwise use defaults
    if selected_weeks and len(selected_weeks) == 3:
        week1_key = str(selected_weeks[0])
        week2_key = str(selected_weeks[1])
        week3_key = str(selected_weeks[2])
        # Validate selected weeks exist
        for wk in [week1_key, week2_key, week3_key]:
            if wk not in schedules:
                return jsonify({"error": f"⚠️ Week {wk} not available. Available weeks: {available_weeks}"})
    else:
        # Default: use first 3 weeks from selected_gw1
        selected_gw1 = session.get("selected_gw1", available_weeks[0])
        try:
            start_idx = available_weeks.index(selected_gw1)
        except ValueError:
            start_idx = 0
        
        if start_idx + 2 >= len(available_weeks):
            start_idx = max(0, len(available_weeks) - 3)
        
        week1_key = available_weeks[start_idx]
        week2_key = available_weeks[start_idx + 1] if start_idx + 1 < len(available_weeks) else available_weeks[-1]
        week3_key = available_weeks[start_idx + 2] if start_idx + 2 < len(available_weeks) else available_weeks[-1]
    
    schedule_week1 = schedules.get(week1_key, {})
    schedule_week2 = schedules.get(week2_key, {})
    schedule_week3 = schedules.get(week3_key, {})
    
    # Convert string keys to int keys if needed
    if schedule_week1 and isinstance(list(schedule_week1.keys())[0] if schedule_week1.keys() else None, str):
        schedule_week1 = {int(k): v for k, v in schedule_week1.items()}
    if schedule_week2 and isinstance(list(schedule_week2.keys())[0] if schedule_week2.keys() else None, str):
        schedule_week2 = {int(k): v for k, v in schedule_week2.items()}
    if schedule_week3 and isinstance(list(schedule_week3.keys())[0] if schedule_week3.keys() else None, str):
        schedule_week3 = {int(k): v for k, v in schedule_week3.items()}
    
    print(f"   Using weeks: {week1_key}, {week2_key}, {week3_key}")
    
    try:
        start_time = time.time()
        
        # Create wildcard optimizer
        optimizer = WildcardOptimizer(
            all_players_df=full_players_df.copy(),
            schedule_week1=schedule_week1,
            schedule_week2=schedule_week2,
            schedule_week3=schedule_week3,
            user_team_df=user_team_df
        )
        
        # Find best teams
        results = optimizer.find_best_wildcard_teams(
            budget=budget,
            top_n=top_n,
            must_include=must_include,
            must_exclude=must_exclude,
            injured=injured,
            w1=w1,
            w2=w2,
            w3=w3
        )
        
        end_time = time.time()
        print(f"✅ Wildcard optimization completed in {end_time - start_time:.2f}s")
        
        if not results:
            return jsonify({"error": "⚠️ No valid teams found with the given constraints."})
        
        # Convert results for JSON response
        response_data = {
            "top_teams": [],
            "weeks_used": [week1_key, week2_key, week3_key],
            "weights_used": {"w1": w1, "w2": w2, "w3": w3}
        }
        
        for result in results:
            team_data = {
                "weighted_score": result['weighted_score'],
                "week1_score": result['week1_score'],
                "week2_score": result['week2_score'],
                "week3_score": result['week3_score'],
                "team_price": result['team_price'],
                "wallet_remaining": result['wallet_remaining'],
                "team": result['team'],
                "week1_schedule": convert_floats(result['week1_schedule']),
                "week2_schedule": convert_floats(result['week2_schedule']),
                "week3_schedule": convert_floats(result['week3_schedule'])
            }
            response_data["top_teams"].append(team_data)
        
        gc.collect()
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"⚠️ Error during wildcard optimization: {str(e)}"})


@app.route("/print_weekly_schedule", methods=["POST"])
def print_weekly_schedule():
    user_team_data = session.get("user_team")

    if not user_team_data:
        return jsonify({"error": "⚠️ No team has been entered yet!"})

    schedule_week1 = session.get("schedule_week1")
    selected_gw1 = session.get("selected_gw1")
    
    # Fallback to first week from SCHEDULE_DATA if session schedule is not set
    if schedule_week1 is None:
        schedules = app.config["SCHEDULE_DATA"]
        available_weeks = sorted(schedules.keys(), key=lambda x: int(x))
        if available_weeks:
            schedule_week1 = schedules[available_weeks[0]]
            selected_gw1 = available_weeks[0]
            print(f"⚠️ print_weekly_schedule using default week: {selected_gw1}")
        else:
            return jsonify({"error": "⚠️ No schedule data available."})
    
    # Convert string keys to int keys if needed
    if schedule_week1 and isinstance(list(schedule_week1.keys())[0] if schedule_week1.keys() else None, str):
        schedule_week1 = {int(k): v for k, v in schedule_week1.items()}

    user_team = pd.DataFrame(user_team_data)
    optimizer = FantasyOptimizer(
        user_team,
        full_players_df,
        schedule_week1,
        schedule_week1
    )
    weekly_schedule = optimizer.print_weekly_form()

    return jsonify({
        "weekly_schedule": weekly_schedule,
        "week_number": selected_gw1 or "N/A"
    })

if __name__ == "__main__":
    from waitress import serve  # Use Waitress instead of Gunicorn for better performance
    serve(app, host="0.0.0.0", port=5000, threads=4)