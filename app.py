from flask import Flask, render_template, request, jsonify, session
from flask_session import Session  # Import Flask-Session
import pandas as pd
import numpy as np
import json
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder
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
schedule_payload = data_load.get_or_update_schedule(num_weeks=3)
app.config["SCHEDULE_DATA"] = schedule_payload["weeks"]
#data_load.load_schedule()

## changing temporarily for showing my code
FULL_PLAYERS_CSV = "data/top_update_players_2026.csv"
## FULL_PLAYERS_CSV = "data/top_update_players(1).csv"
#SCHEDULE_CSV = "data/GWSchedule26_11.csv"
#SECOND_SCHEDULE_CSV ="data/GWSchedule26_11.csv"
BIWEEKLY_SCHEDULE_CSV = "data/GWSchedule26_4.csv"

full_players_df = pd.read_csv(FULL_PLAYERS_CSV)

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
            #player["$"] = row_data.get("$", "unknown")
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
        return jsonify({"error": "Selected week not available"}), 400

    session["schedule_week1"] = schedules[gw1]
    session["schedule_week2"] = schedules[gw2]

    return jsonify({"message": "✅ Week selection saved"})


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
    user_team = user_team[["Player", "$", "Form", "TP.", "Pos","team"]]
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
        sub_day = data.get("sub_day")
        if sub_day:
            sub_day = int(sub_day)  # Convert to int (1-7)
        
        schedule_week1 = session.get("schedule_week1")
        schedule_week2 = session.get("schedule_week2")
        
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
        elif sub_type == "total":
            best_swaps = optimizer.find_best_total_substitutions(
                extra_salary, top_n,
                untradable_players=untradable_players,
                must_trade_players=must_trade_players,
                sub_day=sub_day
            )
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
        elif sub_type == "total":
            response_data["top_substitutions"] = [
                {
                    "form_gain": swap[0],
                    "new_form": swap[1],
                    "current_form": swap[2],
                    "new_salary": swap[3],
                    "substitutions_out": swap[4],
                    "substitutions_in": swap[5],
                    "new_team": clean_dict_keys(swap[6]),
                }
                for swap in best_swaps
            ]
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

@app.route("/print_weekly_schedule", methods=["POST"])
def print_weekly_schedule():
    user_team_data = session.get("user_team")

    if not user_team_data:
        return jsonify({"error": "⚠️ No team has been entered yet!"})

    schedule_week1 = session.get("schedule_week1")
    
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

    return jsonify({"weekly_schedule": weekly_schedule})

if __name__ == "__main__":
    from waitress import serve  # Use Waitress instead of Gunicorn for better performance
    serve(app, host="0.0.0.0", port=5000, threads=4)