from flask import Flask, render_template, request, jsonify
import pandas as pd
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder

app = Flask(__name__)

# Define file paths (update these paths if needed)
FULL_PLAYERS_CSV = "data/top_update_players.csv"
BEST_FILTER_CSV = "data/top_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"

# Load player dataset for autocomplete
full_players_df = pd.read_csv(FULL_PLAYERS_CSV)


@app.route("/")
def home():
    """Render the homepage with player selection."""
    return render_template("index.html")


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """Return player names for autocomplete based on user input."""
    query = request.args.get("query", "").lower()
    suggestions = full_players_df[full_players_df["Player"].str.lower().str.contains(query, na=False)]
    return jsonify(suggestions["Player"].tolist())


@app.route("/find_best_team", methods=["POST"])
def find_best_team():
    """Find the best team using Dynamic Programming"""
    data = request.json
    salary_cap = float(data.get("salary_cap", 100))

    team_finder = BestTeamFinder(FULL_PLAYERS_CSV, salary_cap)
    best_team = team_finder.find_best_team()

    if best_team is not None and not best_team.empty:
        return jsonify({
            "best_team": best_team.to_dict(orient="records"),
            "total_form": best_team["Form"].sum(),
            "total_price": best_team["$"].sum(),
        })
    else:
        return jsonify({"error": "No valid team found within the given salary cap."})


@app.route("/find_best_substitutions", methods=["POST"])
def find_best_substitutions():
    """Find the best substitutions for a user's team"""
    data = request.json
    user_team_data = pd.DataFrame(data.get("user_team"))
    extra_salary = float(data.get("extra_salary", 0))

    optimizer = FantasyOptimizer(user_team_data, FULL_PLAYERS_CSV, SCHEDULE_CSV)

    if data.get("sub_type") == "weekly":
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
