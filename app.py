from flask import Flask, render_template, request, jsonify, session
import pandas as pd
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder

app = Flask(__name__, template_folder="templates")
app.secret_key = "super_secret_key"  # Needed for session storage

# Define file paths
FULL_PLAYERS_CSV = "data/top_update_players.csv"
BEST_FILTER_CSV = "data/top_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"

# Load player dataset for autocomplete
full_players_df = pd.read_csv(FULL_PLAYERS_CSV)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """Return player names for autocomplete based on user input."""
    query = request.args.get("query", "").lower()
    suggestions = full_players_df[full_players_df["Player"].str.lower().str.contains(query, na=False)]
    return jsonify(suggestions["Player"].tolist())


@app.route("/set_user_team", methods=["POST"])
def set_user_team():
    """Store the user's team and extra salary in session storage."""
    data = request.json
    session["user_team"] = data.get("user_team", [])
    session["extra_salary"] = float(data.get("extra_salary", 0))
    
    return jsonify({"message": "User team saved successfully."})


@app.route("/find_best_substitutions", methods=["POST"])
def find_best_substitutions():
    """Find the best substitutions using stored user team."""
    sub_type = request.json.get("sub_type", "weekly")

    # Retrieve stored team and salary
    user_team_data = session.get("user_team", [])
    extra_salary = float(session.get("extra_salary", 0))

    if not user_team_data:
        return jsonify({"error": "No user team found. Please enter your team first."})

    user_team_df = pd.DataFrame(user_team_data)

    optimizer = FantasyOptimizer(user_team_df, FULL_PLAYERS_CSV, SCHEDULE_CSV)

    if sub_type == "weekly":
        best_team, new_form, _, best_out, best_in = optimizer.find_best_weekly_substitutions(extra_salary)
    else:
        best_team, new_form, _, best_out, best_in = optimizer.find_best_total_substitutions(extra_salary)

    return jsonify({
        "new_team": best_team.to_dict(orient="records"),
        "new_form": new_form,
        "substitutions_out": best_out,
        "substitutions_in": best_in
    })


if __name__ == "__main__":
    app.run(debug=True)
