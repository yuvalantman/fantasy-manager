from flask import Flask, render_template, request, jsonify, session
from flask_session import Session  # Import Flask-Session
import pandas as pd
import numpy as np
import json
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder
from data_loader import data_loader

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

FULL_PLAYERS_CSV = "data/top_update_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"

full_players_df = pd.read_csv(FULL_PLAYERS_CSV)

def replace_nan_with_none(obj):
    """Recursively replace NaN values with None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None  # JSON does not support NaN, so we replace it with None
    return obj

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

@app.route("/set_user_team", methods=["POST"])
def set_user_team():
    data = request.get_json()
    user_team = data.get("user_team", [])
    extra_salary = float(data.get("extra_salary", 0))
    updated_players_df = full_players_df.copy()

    for player in user_team:
        updated_players_df.loc[updated_players_df["Player"] == player["Player"], "$"] = player["$"]

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
            player["$"] = row_data.get("$", "unknown")
            player["TP."] = row_data.get("TP.", "unknown")
            player["team"] = row_data.get("team", "unknown")

    
    # Save user team in session
    session["updated_players_df"] = updated_players_df
    session["user_team"] = user_team
    session["extra_salary"] = extra_salary
    session.modified=True

    return jsonify({"message": "✅ User team saved successfully!"})

@app.route("/compute_result", methods=["POST"])
def compute_result():
    data = request.json
    option = data.get("option")
    sub_type = data.get("sub_type", "weekly")
    salary_cap = float(data.get("salary_cap", 100))
    top_n = int(data.get("top_n", 5))
    print(f"📢 Salary Cap Received: {salary_cap}")
    print("📢 Compute Request Received:", data)
    #print("📢 Session Data (user_team):", session.get("user_team"))
    #print("📢 Session Data (updated_players_df):", session.get("updated_players_df"))

    user_team_data = session.get("user_team")
    extra_salary = float(session.get("extra_salary", 0))
    updated_players_data = session.get("updated_players_df")

    if not user_team_data or updated_players_data.empty:
        print("❌ ERROR: User team or updated players missing.")
        return jsonify({"error": "⚠️ User team is missing. Please enter your team first."}), 400

    user_team = pd.DataFrame(user_team_data)
    user_team = user_team[["Player", "$", "Form", "TP.", "Pos","team"]]
    updated_players_df = pd.DataFrame(updated_players_data)
    updated_players_df = updated_players_df[["Player", "$", "Form", "TP.", "Pos","team"]]

    if option == "best_team":
        team_finder = BestTeamFinder(updated_players_df, salary_cap)
        best_team = team_finder.find_best_team()

        if best_team is None or best_team.empty:
            return jsonify({"error": "⚠️ No valid team found within the given salary cap."})

        response = {
            "best_team": best_team.to_dict(orient="records"),
            "total_form": best_team["Form"].sum(),
            "total_price": best_team["$"].sum()
        }
        response = replace_nan_with_none(response)
        print("📢 Sending Response:", response)
        return jsonify(response)

    elif option == "best_substitutions":
        optimizer = FantasyOptimizer(user_team, updated_players_df, SCHEDULE_CSV)

        if sub_type == "weekly":
            best_team, new_form, _, best_out, best_in, weekly_sched = optimizer.find_best_weekly_substitutions(extra_salary)
            response = {
                "new_team": best_team.to_dict(orient="records"),
                "new_form": new_form,
                "substitutions_out": best_out,
                "substitutions_in": best_in,
                "weekly_sched": weekly_sched
            }
            if best_team is None or best_team.empty:
                return jsonify({"error": "⚠️ No valid substitutions found."})
            response = replace_nan_with_none(response)
            print("📢 Sending Response:", response)  # ✅ Debug Flask Response
            return jsonify(response)
        else:
            top_swaps = optimizer.find_best_total_substitutions(extra_salary, top_n)

            if not top_swaps:
                return jsonify({"error": "⚠️ No valid substitutions found."})

            response = {
                "top_substitutions": [
                    {
                        "form_gain": form_gain,
                        "new_team": new_team,
                        "new_form": new_form,
                        "new_salary": new_salary,
                        "substitutions_out": substitutions_out,
                        "substitutions_in": substitutions_in,
                    }
                    for form_gain, new_form, new_salary, substitutions_out, substitutions_in, new_team in top_swaps
                ]
            }
            response = replace_nan_with_none(response)
            print("📢 Sending Response:", response)  # ✅ Debug Flask Response
            return jsonify(response)

    return jsonify({"error": "⚠️ Invalid option selected."})

if __name__ == "__main__":
    app.run(debug=True)
