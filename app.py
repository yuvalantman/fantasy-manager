from flask import Flask, render_template, request, jsonify, session
from flask_session import Session  # Import Flask-Session
import pandas as pd
import json
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder

app = Flask(__name__, template_folder="templates", static_folder="templates/static")
app.secret_key = "super_secret_key"

# Store session on the server instead of cookies
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./.flask_session/"  # Store session files locally
app.config["SECRET_KEY"] = "your_secret_key_here"  # Required for session security
Session(app)  # Initialize server-side session

FULL_PLAYERS_CSV = "data/top_update_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"

full_players_df = pd.read_csv(FULL_PLAYERS_CSV)

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
    print(f"📢 Salary Cap Received: {salary_cap}")
    print("📢 Compute Request Received:", data)
    print("📢 Session Data (user_team):", session.get("user_team"))
    print("📢 Session Data (updated_players_df):", session.get("updated_players_df"))

    user_team_data = session.get("user_team")
    extra_salary = float(session.get("extra_salary", 0))
    updated_players_data = session.get("updated_players_df")

    if not user_team_data or updated_players_data.empty:
        print("❌ ERROR: User team or updated players missing.")
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
