import pandas as pd
from optimizer import FantasyOptimizer
from best_team_finder import BestTeamFinder
from data_loader import data_loader

# Paths to CSV files
FULL_PLAYERS_CSV = "data/top_update_players.csv"  # Full list of players and their actual salaries
BEST_FILTER_CSV = "data/top_players.csv"
SCHEDULE_CSV = "data/GW19Schedule.csv"


def get_user_team():
    """Allows the user to manually input their fantasy team (10 players and salaries)."""
    print("\n📝 Enter Your 10-Player Team")

    # Load the full player dataset
    full_players = pd.read_csv(FULL_PLAYERS_CSV)

    user_team = []

    for i in range(10):
        while True:
            player_name = input(f"Enter Player {i+1} Name: ").strip()

            # Check if the player exists in the database
            matching_players = full_players[full_players["Player"].str.lower() == player_name.lower()]

            if matching_players.empty:
                print("⚠️ Player not found. Please enter a valid player name.")
                continue

            # Ask for salary
            try:
                salary = float(input(f"Enter salary for {player_name}: "))
            except ValueError:
                print("⚠️ Invalid salary. Please enter a number.")
                continue

            
            if player_name.lower() in [player.iloc[1].lower() for player in user_team]:
                print("⚠️ Entered a player that you already have. Please enter a new player")
                continue

            # Fetch player details
            player_data = matching_players.iloc[0].copy()
            player_data["$"] = salary  # Override the salary with user input

            user_team.append(player_data)
            break

    # Convert list to DataFrame
    user_team_df = pd.DataFrame(user_team)

    # Ask if they have extra unused salary
    extra_salary = 0
    use_extra_salary = input("\nDo you have extra salary available? (yes/no): ").strip().lower()
    if use_extra_salary == "yes":
        try:
            extra_salary = float(input("Enter the amount of extra salary: "))
        except ValueError:
            print("⚠️ Invalid amount, using 0 extra salary.")

    return user_team_df, extra_salary


def main1():
    """Main menu to allow the user to interact with the fantasy optimizer."""
    print("\n🏀 Welcome to the Fantasy Team Optimizer!")

    # Initialize the best team finder
    

    while True:
        print("\n🏆 Fantasy Team Menu")
        print("1️⃣ Find the Best Team (DP)")
        print("2️⃣ Find Best Weekly Substitutions for Your Team")
        print("3️⃣ Find Best Total Form Substitutions for Your Team")
        print("4️⃣ Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            # Ask for custom salary cap
            try:
                salary_cap = float(input("\nEnter your max salary cap: "))
            except ValueError:
                print("⚠️ Invalid number, using default salary cap (100)")
                salary_cap = 100
            team_finder = BestTeamFinder(FULL_PLAYERS_CSV,salary_cap)
            best_team = team_finder.find_best_team()

            if best_team is not None and not best_team.empty:
                print(best_team)
                print(f"\n🔥 Best Team Total Form: {best_team['Form'].sum():.2f}")
                print(f"💰 Total Price: {best_team['$'].sum():.2f}")
            else:
                print("⚠️ No valid team found within the given salary cap.")

        elif choice in ["2", "3"]:
            user_team, extra_salary = get_user_team()

            # Initialize the optimizer with the user team
            optimizer = FantasyOptimizer(user_team, FULL_PLAYERS_CSV, SCHEDULE_CSV)

            if choice == "2":
                best_team, new_weekly_form, _, best_out, best_in = optimizer.find_best_weekly_substitutions(extra_salary)
                print("\n🟢 Best Weekly Substitutions:")
                print(best_team)

            elif choice == "3":
                best_team, new_total_form, _, best_out, best_in = optimizer.find_best_total_substitutions(extra_salary)
                print("\n🟢 Best Total Form Substitutions:")
                print(best_team)

        elif choice == "4":
            print("👋 Exiting Fantasy Optimizer. Have a great day!")
            break

        else:
            print("⚠️ Invalid choice, please enter 1, 2, 3, or 4.")

def main():
    d1 = data_loader()
    d1.load_teams()

if __name__ == "__main__":
    main()
