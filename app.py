from flask import Flask, render_template, request, jsonify, session
import pandas as pd
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder

app = Flask(__name__, template_folder="templates", static_folder="templates/static")
app.secret_key = "super_secret_key"

FULL_PLAYERS_CSV = "data/top_update_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"

full_players_df = pd.read_csv(FULL_PLAYERS_CSV)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """Return player names for autocomplete based on user input."""
    query = request.args.get("query", "").lower()
    if not query:
        return jsonify([])

    suggestions = full_players_df[full_players_df["Player"].str.lower().str.contains(query, na=False)]
    return jsonify(suggestions["Player"].tolist())

@app.route("/set_user_team", methods=["POST"])
def set_user_team():
    """Store the user's team and update salaries in the dataset."""
    data = request.json
    user_team = data.get("user_team", [])
    extra_salary = float(data.get("extra_salary", 0))

    if len(user_team) != 10:
        return jsonify({"error": "Please enter exactly 10 players."}), 400

    session["user_team"] = user_team
    session["extra_salary"] = extra_salary

    updated_players_df = full_players_df.copy()
    for player in user_team:
        updated_players_df.loc[updated_players_df["Player"] == player["Player"], "$"] = player["$"]

    session["updated_players_df"] = updated_players_df.to_dict(orient="records")

    return jsonify({"message": "User team saved successfully."})

@app.route("/compute_result", methods=["POST"])
def compute_result():
    """Find the best team or best substitutions based on user choice."""
    data = request.json
    option = data.get("option")
    sub_type = data.get("sub_type", "weekly")
    salary_cap = float(data.get("salary_cap", 100))

    user_team_data = session.get("user_team")
    extra_salary = float(session.get("extra_salary", 0))
    updated_players_data = session.get("updated_players_df")

    if not user_team_data or not updated_players_data:
        return jsonify({"error": "⚠️ User team is missing. Please enter your team first."}), 400

    user_team = pd.DataFrame(user_team_data)
    updated_players_df = pd.DataFrame(updated_players_data)

    if option == "best_team":
        team_finder = BestTeamFinder(updated_players_df, salary_cap)
        best_team = team_finder.find_best_team()

        if best_team is None or best_team.empty:
            return jsonify({"error": "⚠️ No valid team found within the given salary cap."})

        return jsonify({
            "best_team": best_team.to_dict(orient="records"),
            "total_form": best_team["Form"].sum(),
            "total_price": best_team["$"].sum()
        })

    elif option == "best_substitutions":
        optimizer = FantasyOptimizer(user_team, updated_players_df, SCHEDULE_CSV)

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

    return jsonify({"error": "⚠️ Invalid option selected."})

if __name__ == "__main__":
    app.run(debug=True)
