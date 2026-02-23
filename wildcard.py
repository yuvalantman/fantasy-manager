"""
wildcard.py

Wildcard Optimizer for Fantasy Team selection.

This module finds the best possible 10-player lineup from scratch under constraints:
- 5 front and 5 back players
- No more than 2 players from the same team
- Total price must be under budget

Uses weighted weekly form calculation across 3 weeks:
- Week 1: 0.5 weight (default)
- Week 2: 0.3 weight (default)
- Week 3: 0.2 weight (default)

Supports:
- User-defined weights for each week
- Must-include players (locked in)
- Must-exclude players (not considered)
- Injured players (not considered)
- User team player prices used for budget calculation
- Parallel multiprocessing for optimization

Algorithm Overview:
1. Filter players based on constraints (must-include, must-exclude, injured)
2. Calculate weighted 3-week form scores for all players
3. Use dynamic programming with early constraint checking
4. Generate top-N lineup options
5. For each option: show team, weekly scores, budget usage, schedules
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
import time
import heapq
import random


@dataclass
class WildcardPlayerData:
    """Lightweight player data for wildcard optimization."""
    name: str
    pos: str  # 'front' or 'back'
    nba_team: str
    price: float  # price to use (user team price if on team, market otherwise)
    market_price: float  # original market price
    form: float
    week1_score: float = 0.0
    week2_score: float = 0.0
    week3_score: float = 0.0
    weighted_score: float = 0.0
    week1_bitmask: int = 0  # bitmask of days playing in week 1
    week2_bitmask: int = 0
    week3_bitmask: int = 0
    week1_forms: Tuple[float, ...] = field(default_factory=tuple)
    week2_forms: Tuple[float, ...] = field(default_factory=tuple)
    week3_forms: Tuple[float, ...] = field(default_factory=tuple)


class WildcardOptimizer:
    """
    Wildcard Optimizer for finding the best complete team from scratch.
    
    Uses dynamic programming with weighted 3-week form scoring.
    """
    
    def __init__(
        self,
        all_players_df: pd.DataFrame,
        schedule_week1: Dict[int, List[str]],
        schedule_week2: Dict[int, List[str]],
        schedule_week3: Dict[int, List[str]],
        user_team_df: pd.DataFrame = None
    ):
        """
        Initialize wildcard optimizer.
        
        Args:
            all_players_df: DataFrame with all available players
            schedule_week1: Dict mapping day (1-7) to list of playing NBA teams
            schedule_week2: Dict for week 2 schedule
            schedule_week3: Dict for week 3 schedule
            user_team_df: Optional DataFrame with user's current team (for custom prices)
        """
        self.all_players_df = all_players_df.copy()
        self.schedule_week1 = schedule_week1 or {}
        self.schedule_week2 = schedule_week2 or {}
        self.schedule_week3 = schedule_week3 or {}
        
        # Standardize columns
        self._standardize_columns()
        
        # Build user team price lookup
        self.user_team_prices = {}
        if user_team_df is not None and not user_team_df.empty:
            user_team_df = user_team_df.copy()
            if "$" in user_team_df.columns:
                user_team_df = user_team_df.rename(columns={"$": "Salary"})
            for _, row in user_team_df.iterrows():
                self.user_team_prices[row["Player"]] = row["Salary"]
        
        # Build player data cache
        self._player_cache: Dict[str, WildcardPlayerData] = {}
        
    def _standardize_columns(self):
        """Standardize column names and data types."""
        if "$" in self.all_players_df.columns and "Salary" not in self.all_players_df.columns:
            self.all_players_df = self.all_players_df.rename(columns={"$": "Salary"})
        
        self.all_players_df["Pos"] = self.all_players_df["Pos"].str.lower()
        self.all_players_df["Salary"] = self.all_players_df["Salary"].astype(float)
        self.all_players_df["Form"] = self.all_players_df["Form"].astype(float)
        
    def _get_day_bitmask_and_forms(
        self, 
        nba_team: str, 
        form: float, 
        schedule: Dict[int, List[str]]
    ) -> Tuple[int, Tuple[float, ...]]:
        """Calculate bitmask and daily forms for a week."""
        bitmask = 0
        day_forms = []
        for day in range(1, 8):
            playing_teams = schedule.get(day, [])
            if nba_team in playing_teams:
                bitmask |= (1 << (day - 1))
                day_forms.append(form)
            else:
                day_forms.append(0.0)
        return bitmask, tuple(day_forms)
    
    def _compute_weekly_lineup_score(
        self, 
        players: List[WildcardPlayerData], 
        week_forms_getter
    ) -> float:
        """
        Compute total score for a week using daily lineup optimization.
        
        Args:
            players: List of player data
            week_forms_getter: Function to get day forms for a player
            
        Returns:
            Total weekly score
        """
        total_score = 0.0
        
        for day in range(1, 8):
            front_forms = []
            back_forms = []
            
            for p in players:
                forms = week_forms_getter(p)
                if forms[day - 1] > 0:
                    if p.pos == "front":
                        front_forms.append(forms[day - 1])
                    else:
                        back_forms.append(forms[day - 1])
            
            # Get top 3 from each position
            front_forms.sort(reverse=True)
            back_forms.sort(reverse=True)
            
            top_front = front_forms[:3]
            top_back = back_forms[:3]
            
            # Combine and take top 5
            combined = top_front + top_back
            combined.sort(reverse=True)
            
            total_score += sum(combined[:5])
            
        return total_score
    
    def _build_player_data(
        self,
        must_include: List[str],
        must_exclude: List[str],
        injured: List[str],
        w1: float,
        w2: float,
        w3: float
    ) -> List[WildcardPlayerData]:
        """Build player data list with filtering and scoring."""
        excluded_names = set(must_exclude) | set(injured)
        seen_player_names = set()  # Track seen players to avoid duplicates
        
        players = []
        for _, row in self.all_players_df.iterrows():
            name = row["Player"]
            
            # Skip duplicate player entries
            if name in seen_player_names:
                continue
            seen_player_names.add(name)
            
            # Skip excluded players
            if name in excluded_names:
                continue
                
            nba_team = row.get("team", "")
            form = row["Form"]
            market_price = row["Salary"]
            
            # Use user team price if available
            price = self.user_team_prices.get(name, market_price)
            
            # Handle missing position
            pos_raw = row.get("Pos", "")
            if pd.isna(pos_raw) or pos_raw == "":
                pos = "back"
            else:
                pos = str(pos_raw).lower()
            
            # Calculate week bitmasks and daily forms
            w1_bitmask, w1_forms = self._get_day_bitmask_and_forms(
                nba_team, form, self.schedule_week1
            )
            w2_bitmask, w2_forms = self._get_day_bitmask_and_forms(
                nba_team, form, self.schedule_week2
            )
            w3_bitmask, w3_forms = self._get_day_bitmask_and_forms(
                nba_team, form, self.schedule_week3
            )
            
            # Calculate weekly scores (sum of playing days)
            week1_score = sum(w1_forms)
            week2_score = sum(w2_forms)
            week3_score = sum(w3_forms)
            
            # Calculate weighted score
            weighted_score = w1 * week1_score + w2 * week2_score + w3 * week3_score
            
            player_data = WildcardPlayerData(
                name=name,
                pos=pos,
                nba_team=nba_team,
                price=price,
                market_price=market_price,
                form=form,
                week1_score=week1_score,
                week2_score=week2_score,
                week3_score=week3_score,
                weighted_score=weighted_score,
                week1_bitmask=w1_bitmask,
                week2_bitmask=w2_bitmask,
                week3_bitmask=w3_bitmask,
                week1_forms=w1_forms,
                week2_forms=w2_forms,
                week3_forms=w3_forms
            )
            
            players.append(player_data)
            self._player_cache[name] = player_data
            
        return players
    
    def _check_team_valid(
        self,
        team: List[WildcardPlayerData],
        max_price: float
    ) -> bool:
        """Check if team satisfies all constraints."""
        if len(team) != 10:
            return False
            
        front_count = sum(1 for p in team if p.pos == "front")
        back_count = sum(1 for p in team if p.pos == "back")
        
        if front_count != 5 or back_count != 5:
            return False
            
        # Check team constraint
        team_counts = defaultdict(int)
        for p in team:
            team_counts[p.nba_team] += 1
            if team_counts[p.nba_team] > 2:
                return False
                
        # Check price constraint
        total_price = sum(p.price for p in team)
        if total_price > max_price:
            return False
            
        return True
    
    def _knapsack_dp_with_team_constraint(
        self,
        players: List[WildcardPlayerData],
        k_players: int,
        max_price_int: int,
        score_key: str = "weighted_score"
    ) -> Tuple[np.ndarray, dict]:
        """
        DP knapsack with team constraint check during backtracking.
        
        Returns:
            dp: DP table dp[k][w] = max score for k players with cost w
            choices: backtracking info
        """
        n = len(players)
        dp = np.full((k_players + 1, max_price_int + 1), -np.inf)
        dp[0, :] = 0
        choices = {}
        
        for i, player in enumerate(players):
            cost = int(player.price * 10)
            score = getattr(player, score_key)
            
            # Iterate in reverse to avoid using same player twice
            for k in range(k_players, 0, -1):
                for w in range(max_price_int, cost - 1, -1):
                    if dp[k - 1, w - cost] > -np.inf:
                        new_score = dp[k - 1, w - cost] + score
                        if new_score > dp[k, w]:
                            dp[k, w] = new_score
                            choices[(k, w)] = (player.name, k - 1, w - cost)
                            
        return dp, choices
    
    def _backtrack_team(
        self,
        players: List[WildcardPlayerData],
        choices: dict,
        k: int,
        w: int
    ) -> List[WildcardPlayerData]:
        """Reconstruct team from DP choices."""
        player_lookup = {p.name: p for p in players}
        team = []
        seen_names = set()  # Track seen players to avoid duplicates
        
        while (k, w) in choices:
            player_name, prev_k, prev_w = choices[(k, w)]
            # Only add if not already in team (prevents duplicates from DP path issues)
            if player_name not in seen_names:
                team.append(player_lookup[player_name])
                seen_names.add(player_name)
            k, w = prev_k, prev_w
            
        return team
    
    def _repair_team_constraint(
        self,
        team: List[WildcardPlayerData],
        available_pool: List[WildcardPlayerData],
        max_price: float,
        must_include: List[str] = None
    ) -> List[WildcardPlayerData]:
        """
        Repair team constraint violations (max 2 per NBA team).
        Uses heuristic replacement with best available players.
        Never removes must-include players.
        """
        if must_include is None:
            must_include = []
        must_include_names = set(must_include)
        
        repaired = list(team)
        
        while True:
            team_counts = defaultdict(int)
            for p in repaired:
                team_counts[p.nba_team] += 1
                
            # Find violating teams
            violating_teams = {t: c for t, c in team_counts.items() if c > 2}
            
            if not violating_teams:
                break
                
            # Fix first violating team
            team_to_fix = list(violating_teams.keys())[0]
            
            # Get players from violating team, sorted by weighted_score ascending
            # NEVER include must-include players in removable list
            violating_players = sorted(
                [p for p in repaired if p.nba_team == team_to_fix and p.name not in must_include_names],
                key=lambda x: x.weighted_score
            )
            
            # Remove lowest scoring player(s) from this team
            num_to_remove = violating_teams[team_to_fix] - 2
            players_to_remove = violating_players[:num_to_remove]
            
            repaired = [p for p in repaired if p not in players_to_remove]
            
            # Calculate current team salary and constraints
            current_salary = sum(p.price for p in repaired)
            salary_cap_remaining = max_price - current_salary
            
            # Get teams that can still receive players
            current_team_counts = defaultdict(int)
            for p in repaired:
                current_team_counts[p.nba_team] += 1
            valid_teams_for_add = {t for t, c in current_team_counts.items() if c < 2}
            
            # Add all teams not yet on roster
            all_nba_teams = {p.nba_team for p in available_pool}
            teams_not_on_roster = all_nba_teams - set(current_team_counts.keys())
            valid_teams_for_add = valid_teams_for_add | teams_not_on_roster
            
            repaired_names = {p.name for p in repaired}
            
            for player_out in players_to_remove:
                pos_needed = player_out.pos
                
                # Find valid candidates
                candidates = [
                    p for p in available_pool
                    if p.name not in repaired_names
                    and p.pos == pos_needed
                    and p.price <= salary_cap_remaining
                    and p.nba_team in valid_teams_for_add
                ]
                
                if candidates:
                    # Sort by weighted score
                    candidates.sort(key=lambda x: x.weighted_score, reverse=True)
                    best = candidates[0]
                    repaired.append(best)
                    repaired_names.add(best.name)
                    salary_cap_remaining -= best.price
                    current_team_counts[best.nba_team] += 1
                    if current_team_counts[best.nba_team] >= 2:
                        valid_teams_for_add.discard(best.nba_team)
                        
        return repaired
    
    def _knapsack_dp_with_custom_scores(
        self,
        players: List[WildcardPlayerData],
        k_players: int,
        max_price_int: int,
        score_func: Callable[[WildcardPlayerData], float]
    ) -> Tuple[np.ndarray, dict]:
        """
        DP knapsack with custom scoring function.
        
        Args:
            players: List of players
            k_players: Number of players to select
            max_price_int: Budget in integer units (price * 10)
            score_func: Function to compute score for each player
            
        Returns:
            dp: DP table dp[k][w] = max score for k players with cost w
            choices: backtracking info
        """
        dp = np.full((k_players + 1, max_price_int + 1), -np.inf)
        dp[0, :] = 0
        choices = {}
        
        for i, player in enumerate(players):
            cost = int(player.price * 10)
            score = score_func(player)
            
            # Iterate in reverse to avoid using same player twice
            for k in range(k_players, 0, -1):
                for w in range(max_price_int, cost - 1, -1):
                    if dp[k - 1, w - cost] > -np.inf:
                        new_score = dp[k - 1, w - cost] + score
                        if new_score > dp[k, w]:
                            dp[k, w] = new_score
                            choices[(k, w)] = (player.name, k - 1, w - cost)
                            
        return dp, choices

    def _generate_candidate_teams(
        self,
        fronts: List[WildcardPlayerData],
        backs: List[WildcardPlayerData],
        all_players: List[WildcardPlayerData],
        must_include: List[str],
        max_price: float,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        candidate_pool_size: int = 300,
        num_perturbations: int = 30,
        perturbation_strength: float = 0.7,
        num_exclusion_runs: int = 20,
        exclusion_count: int = 3
    ) -> List[List[WildcardPlayerData]]:
        """
        Generate diverse candidate teams using multiple DP strategies.
        
        Strategies:
        1. Multiple scoring objectives (weighted, week1, week2, week3, form, games, value)
        2. Strong random perturbations to break ties and explore nearby solutions
        3. Exclusion runs - randomly exclude top players to force alternatives
        4. Union all unique teams from all strategies
        """
        max_price_int = int(max_price * 10)
        
        # Handle must-include players
        must_include_fronts = [p for p in all_players if p.name in must_include and p.pos == "front"]
        must_include_backs = [p for p in all_players if p.name in must_include and p.pos == "back"]
        
        num_must_fronts = len(must_include_fronts)
        num_must_backs = len(must_include_backs)
        
        # Calculate price already committed to must-include
        must_include_price = sum(p.price for p in must_include_fronts + must_include_backs)
        remaining_price_int = max_price_int - int(must_include_price * 10)
        
        # Calculate how many more players needed
        fronts_needed = 5 - num_must_fronts
        backs_needed = 5 - num_must_backs
        
        if fronts_needed < 0 or backs_needed < 0:
            print(f"⚠️ Too many must-include players: {num_must_fronts} fronts, {num_must_backs} backs")
            return []
            
        # Filter out must-include from available pool
        must_include_names = set(must_include)
        available_fronts = [p for p in fronts if p.name not in must_include_names]
        available_backs = [p for p in backs if p.name not in must_include_names]
        
        # Define multiple scoring strategies
        def weighted_score(p): return p.weighted_score
        def week1_score(p): return p.week1_score
        def week2_score(p): return p.week2_score
        def week3_score(p): return p.week3_score
        def form_score(p): return p.form
        def games_week1(p): return bin(p.week1_bitmask).count('1')
        def games_week2(p): return bin(p.week2_bitmask).count('1')
        def games_week3(p): return bin(p.week3_bitmask).count('1')
        def total_games(p): return games_week1(p) + games_week2(p) + games_week3(p)
        def form_x_games(p): return p.form * total_games(p)
        # Value-based: score per dollar
        def value_score(p): return p.weighted_score / max(p.price, 0.1)
        def week1_value(p): return p.week1_score / max(p.price, 0.1)
        def week2_value(p): return p.week2_score / max(p.price, 0.1)
        def week3_value(p): return p.week3_score / max(p.price, 0.1)
        
        # Weight-specific scoring that heavily emphasizes the most weighted week
        def weight_biased_w1(p): return w1 * 2.0 * p.week1_score + w2 * 0.5 * p.week2_score + w3 * 0.5 * p.week3_score
        def weight_biased_w2(p): return w1 * 0.5 * p.week1_score + w2 * 2.0 * p.week2_score + w3 * 0.5 * p.week3_score
        def weight_biased_w3(p): return w1 * 0.5 * p.week1_score + w2 * 0.5 * p.week2_score + w3 * 2.0 * p.week3_score
        
        # Extreme focus strategies for each week (pure week focus)
        def pure_week1(p): return p.week1_score * 10 + p.week2_score + p.week3_score
        def pure_week2(p): return p.week1_score + p.week2_score * 10 + p.week3_score
        def pure_week3(p): return p.week1_score + p.week2_score + p.week3_score * 10
        
        # Create perturbation score functions with STRONG noise
        def make_perturbed_score(base_func, strength, seed):
            rng = random.Random(seed)
            noise_cache = {}
            def perturbed(p):
                if p.name not in noise_cache:
                    base = base_func(p)
                    # Add substantial noise to break dominance of top players
                    noise = rng.gauss(0, abs(base) * strength + 1.0)
                    noise_cache[p.name] = base + noise
                return noise_cache[p.name]
            return perturbed
        
        # All scoring strategies to try
        base_strategies = [
            ("weighted", weighted_score),
            ("week1", week1_score),
            ("week2", week2_score),
            ("week3", week3_score),
            ("form", form_score),
            ("total_games", total_games),
            ("form_x_games", form_x_games),
            ("value", value_score),
            ("week1_value", week1_value),
            ("week2_value", week2_value),
            ("week3_value", week3_value),
            # Weight-biased strategies
            ("weight_biased_w1", weight_biased_w1),
            ("weight_biased_w2", weight_biased_w2),
            ("weight_biased_w3", weight_biased_w3),
            # Pure week focus strategies
            ("pure_week1", pure_week1),
            ("pure_week2", pure_week2),
            ("pure_week3", pure_week3),
        ]
        
        # Add perturbed versions with varying strengths
        perturbation_strategies = []
        for i in range(num_perturbations):
            # Vary the strength to get different levels of exploration
            strength = perturbation_strength * (0.5 + (i % 5) * 0.3)  # 0.35 to 0.95
            perturbation_strategies.append(
                (f"perturbed_{i}", make_perturbed_score(weighted_score, strength, seed=i*42))
            )
        
        # Collect all candidate teams
        all_candidate_teams = []
        seen_teams = set()
        
        def run_dp_and_collect(fronts_pool, backs_pool, score_func, strategy_name):
            """Run DP on given pools and collect unique teams."""
            nonlocal all_candidate_teams, seen_teams
            
            if len(fronts_pool) < fronts_needed or len(backs_pool) < backs_needed:
                return 0
            
            dp_front, choices_front = self._knapsack_dp_with_custom_scores(
                fronts_pool, fronts_needed, remaining_price_int, score_func
            )
            dp_back, choices_back = self._knapsack_dp_with_custom_scores(
                backs_pool, backs_needed, remaining_price_int, score_func
            )
            
            # Find valid budget splits
            valid_splits = []
            for w_front in range(remaining_price_int + 1):
                w_back = remaining_price_int - w_front
                if w_back >= 0:
                    if dp_front[fronts_needed, w_front] > -np.inf and dp_back[backs_needed, w_back] > -np.inf:
                        total_score = dp_front[fronts_needed, w_front] + dp_back[backs_needed, w_back]
                        valid_splits.append({
                            'score': total_score,
                            'w_front': w_front,
                            'w_back': w_back
                        })
            
            valid_splits.sort(key=lambda x: x['score'], reverse=True)
            
            teams_added = 0
            for split in valid_splits[:30]:  # Top 30 splits per strategy
                w_front, w_back = split['w_front'], split['w_back']
                
                team_fronts = self._backtrack_team(fronts_pool, choices_front, fronts_needed, w_front)
                team_backs = self._backtrack_team(backs_pool, choices_back, backs_needed, w_back)
                
                if len(team_fronts) == fronts_needed and len(team_backs) == backs_needed:
                    full_team = must_include_fronts + must_include_backs + team_fronts + team_backs
                    
                    # Deduplicate
                    seen_in_team = set()
                    deduped_team = []
                    for p in full_team:
                        if p.name not in seen_in_team:
                            deduped_team.append(p)
                            seen_in_team.add(p.name)
                    full_team = deduped_team
                    
                    if len(full_team) != 10:
                        continue
                    
                    repaired_team = self._repair_team_constraint(full_team, all_players, max_price, must_include)
                    
                    repaired_names = [p.name for p in repaired_team]
                    if len(repaired_names) != len(set(repaired_names)):
                        continue
                    
                    # Verify must-include players are still present
                    repaired_names_set = set(repaired_names)
                    if not all(name in repaired_names_set for name in must_include):
                        continue
                    
                    team_key = tuple(sorted(p.name for p in repaired_team))
                    
                    if team_key not in seen_teams and len(repaired_team) == 10:
                        all_candidate_teams.append(repaired_team)
                        seen_teams.add(team_key)
                        teams_added += 1
            
            return teams_added
        
        # Phase 1: Run all base + perturbation strategies on full pool
        all_strategies = base_strategies + perturbation_strategies
        print(f"   Phase 1: Running {len(all_strategies)} scoring strategies...")
        
        for strategy_name, score_func in all_strategies:
            run_dp_and_collect(available_fronts, available_backs, score_func, strategy_name)
        
        phase1_count = len(all_candidate_teams)
        print(f"   Phase 1 complete: {phase1_count} unique teams")
        
        # Phase 2: Exclusion runs - exclude top players to force alternatives
        print(f"   Phase 2: Running {num_exclusion_runs} exclusion strategies...")
        
        # Sort players by weighted score to identify top players
        sorted_fronts = sorted(available_fronts, key=lambda p: p.weighted_score, reverse=True)
        sorted_backs = sorted(available_backs, key=lambda p: p.weighted_score, reverse=True)
        
        rng = random.Random(12345)
        for i in range(num_exclusion_runs):
            # Randomly exclude some top players (from top 15)
            top_front_count = min(15, len(sorted_fronts))
            top_back_count = min(15, len(sorted_backs))
            
            # Exclude 2-4 random top fronts
            num_exclude_front = rng.randint(2, min(4, top_front_count))
            excluded_front_indices = set(rng.sample(range(top_front_count), num_exclude_front))
            
            # Exclude 2-4 random top backs  
            num_exclude_back = rng.randint(2, min(4, top_back_count))
            excluded_back_indices = set(rng.sample(range(top_back_count), num_exclude_back))
            
            # Create filtered pools
            filtered_fronts = [p for j, p in enumerate(sorted_fronts) if j not in excluded_front_indices]
            filtered_backs = [p for j, p in enumerate(sorted_backs) if j not in excluded_back_indices]
            
            # Run DP with weighted score on filtered pool
            run_dp_and_collect(filtered_fronts, filtered_backs, weighted_score, f"exclusion_{i}")
            
            # Also run with a perturbed score on filtered pool for more variety
            perturbed_func = make_perturbed_score(weighted_score, 0.5, seed=i*100+7)
            run_dp_and_collect(filtered_fronts, filtered_backs, perturbed_func, f"exclusion_perturbed_{i}")
        
        phase2_count = len(all_candidate_teams) - phase1_count
        print(f"   Phase 2 complete: {phase2_count} additional teams")
        
        # Phase 3: Random team construction for more diversity
        print(f"   Phase 3: Generating random valid teams...")
        
        random_teams_target = 100
        random_attempts = 0
        max_random_attempts = 1000
        
        while len(all_candidate_teams) - phase1_count - phase2_count < random_teams_target and random_attempts < max_random_attempts:
            random_attempts += 1
            
            # Probabilistic selection weighted by score
            front_weights = [max(0.1, p.weighted_score) for p in available_fronts]
            back_weights = [max(0.1, p.weighted_score) for p in available_backs]
            
            # Normalize weights
            front_total = sum(front_weights)
            back_total = sum(back_weights)
            front_probs = [w/front_total for w in front_weights]
            back_probs = [w/back_total for w in back_weights]
            
            # Sample players (with replacement initially, then dedupe)
            try:
                sampled_front_indices = rng.choices(range(len(available_fronts)), weights=front_probs, k=fronts_needed * 2)
                sampled_back_indices = rng.choices(range(len(available_backs)), weights=back_probs, k=backs_needed * 2)
                
                # Dedupe while maintaining order
                seen_f = set()
                unique_front_indices = []
                for idx in sampled_front_indices:
                    if idx not in seen_f and len(unique_front_indices) < fronts_needed:
                        unique_front_indices.append(idx)
                        seen_f.add(idx)
                
                seen_b = set()
                unique_back_indices = []
                for idx in sampled_back_indices:
                    if idx not in seen_b and len(unique_back_indices) < backs_needed:
                        unique_back_indices.append(idx)
                        seen_b.add(idx)
                
                if len(unique_front_indices) < fronts_needed or len(unique_back_indices) < backs_needed:
                    continue
                
                team_fronts = [available_fronts[i] for i in unique_front_indices]
                team_backs = [available_backs[i] for i in unique_back_indices]
                
                full_team = must_include_fronts + must_include_backs + team_fronts + team_backs
                
                # Check budget
                total_price = sum(p.price for p in full_team)
                if total_price > max_price:
                    continue
                
                # Deduplicate
                seen_in_team = set()
                deduped_team = []
                for p in full_team:
                    if p.name not in seen_in_team:
                        deduped_team.append(p)
                        seen_in_team.add(p.name)
                
                if len(deduped_team) != 10:
                    continue
                
                repaired_team = self._repair_team_constraint(deduped_team, all_players, max_price, must_include)
                
                repaired_names = [p.name for p in repaired_team]
                if len(repaired_names) != len(set(repaired_names)):
                    continue
                
                # Verify must-include players are still present
                repaired_names_set = set(repaired_names)
                if not all(name in repaired_names_set for name in must_include):
                    continue
                
                team_key = tuple(sorted(p.name for p in repaired_team))
                
                if team_key not in seen_teams and len(repaired_team) == 10:
                    all_candidate_teams.append(repaired_team)
                    seen_teams.add(team_key)
                    
            except Exception:
                continue
        
        phase3_count = len(all_candidate_teams) - phase1_count - phase2_count
        print(f"   Phase 3 complete: {phase3_count} random teams")
        
        # Phase 4: Greedy construction starting from top individual players
        print(f"   Phase 4: Greedy team construction with diverse seeds...")
        
        phase4_start = len(all_candidate_teams)
        
        # Sort fronts and backs by different criteria for diverse seeding
        sort_criteria = [
            ("week1", lambda p: p.week1_score),
            ("week2", lambda p: p.week2_score),
            ("week3", lambda p: p.week3_score),
            ("weighted", lambda p: p.weighted_score),
            ("form", lambda p: p.form),
            ("value", lambda p: p.weighted_score / max(p.price, 0.1)),
        ]
        
        for criteria_name, sort_key in sort_criteria:
            sorted_fronts = sorted(available_fronts, key=sort_key, reverse=True)
            sorted_backs = sorted(available_backs, key=sort_key, reverse=True)
            
            # Try building teams starting with top players by this criteria
            for seed_offset in range(0, min(10, len(sorted_fronts) - fronts_needed, len(sorted_backs) - backs_needed)):
                try:
                    team = []
                    team_names = set()
                    team_counts = defaultdict(int)
                    
                    # Add must-includes first
                    for p in must_include_fronts + must_include_backs:
                        team.append(p)
                        team_names.add(p.name)
                        team_counts[p.nba_team] += 1
                    
                    remaining_budget = max_price - sum(p.price for p in team)
                    
                    # Greedily add fronts
                    for p in sorted_fronts[seed_offset:]:
                        if len([t for t in team if t.pos == "front"]) >= 5:
                            break
                        if p.name in team_names:
                            continue
                        if p.price > remaining_budget:
                            continue
                        if team_counts[p.nba_team] >= 2:
                            continue
                        team.append(p)
                        team_names.add(p.name)
                        team_counts[p.nba_team] += 1
                        remaining_budget -= p.price
                    
                    # Greedily add backs
                    for p in sorted_backs[seed_offset:]:
                        if len([t for t in team if t.pos == "back"]) >= 5:
                            break
                        if p.name in team_names:
                            continue
                        if p.price > remaining_budget:
                            continue
                        if team_counts[p.nba_team] >= 2:
                            continue
                        team.append(p)
                        team_names.add(p.name)
                        team_counts[p.nba_team] += 1
                        remaining_budget -= p.price
                    
                    if len(team) == 10:
                        front_count = sum(1 for t in team if t.pos == "front")
                        back_count = sum(1 for t in team if t.pos == "back")
                        if front_count == 5 and back_count == 5:
                            team_key = tuple(sorted(p.name for p in team))
                            if team_key not in seen_teams:
                                all_candidate_teams.append(team)
                                seen_teams.add(team_key)
                                
                except Exception:
                    continue
        
        phase4_count = len(all_candidate_teams) - phase4_start
        print(f"   Phase 4 complete: {phase4_count} greedy teams")
        
        print(f"   Total: {len(all_candidate_teams)} unique candidate teams")
        return all_candidate_teams
    
    def _compute_team_weekly_scores(
        self,
        team: List[WildcardPlayerData]
    ) -> Tuple[float, float, float]:
        """Compute accurate weekly scores using daily lineup optimization."""
        week1_score = self._compute_weekly_lineup_score(
            team, lambda p: p.week1_forms
        )
        week2_score = self._compute_weekly_lineup_score(
            team, lambda p: p.week2_forms
        )
        week3_score = self._compute_weekly_lineup_score(
            team, lambda p: p.week3_forms
        )
        return week1_score, week2_score, week3_score
    
    def _compute_team_weighted_score(
        self,
        team: List[WildcardPlayerData],
        w1: float,
        w2: float,
        w3: float
    ) -> float:
        """Compute weighted score using actual daily lineup optimization."""
        week1, week2, week3 = self._compute_team_weekly_scores(team)
        return w1 * week1 + w2 * week2 + w3 * week3
    
    def _local_search_improve_team(
        self,
        team: List[WildcardPlayerData],
        all_players: List[WildcardPlayerData],
        max_price: float,
        w1: float,
        w2: float,
        w3: float,
        must_include: List[str] = None,
        max_iterations: int = 50,
        max_no_improvement: int = 15
    ) -> List[WildcardPlayerData]:
        """
        Improve a team using local search (hill climbing with swaps).
        
        Uses the actual weighted team score as objective.
        Never swaps out must-include players.
        """
        if must_include is None:
            must_include = []
        must_include_names = set(must_include)
        
        current_team = list(team)
        team_names = {p.name for p in current_team}
        current_score = self._compute_team_weighted_score(current_team, w1, w2, w3)
        
        # Build lookup structures
        player_lookup = {p.name: p for p in all_players}
        
        # Separate candidates by position
        front_candidates = [p for p in all_players if p.pos == "front" and p.name not in team_names]
        back_candidates = [p for p in all_players if p.pos == "back" and p.name not in team_names]
        
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            if no_improvement_count >= max_no_improvement:
                break
                
            improved = False
            
            # Try swapping each player in team with candidates
            # Skip must-include players - they cannot be swapped out
            for i, player_out in enumerate(current_team):
                if player_out.name in must_include_names:
                    continue  # Never swap out must-include players
                pos = player_out.pos
                candidates = front_candidates if pos == "front" else back_candidates
                
                # Get current team constraint status
                team_counts = defaultdict(int)
                for p in current_team:
                    team_counts[p.nba_team] += 1
                
                # Calculate current price without this player
                price_without = sum(p.price for j, p in enumerate(current_team) if j != i)
                max_swap_price = max_price - price_without
                
                # Try each candidate
                for candidate in candidates:
                    # Check price constraint
                    if candidate.price > max_swap_price:
                        continue
                    
                    # Check team constraint
                    # After removing player_out, can we add candidate?
                    new_team_count = team_counts[candidate.nba_team]
                    if candidate.nba_team == player_out.nba_team:
                        new_team_count -= 1  # We're removing one from same team
                    if new_team_count >= 2:
                        continue
                    
                    # Create new team
                    new_team = [p for j, p in enumerate(current_team) if j != i]
                    new_team.append(candidate)
                    
                    # Evaluate
                    new_score = self._compute_team_weighted_score(new_team, w1, w2, w3)
                    
                    if new_score > current_score + 0.1:  # Small threshold to avoid noise
                        current_team = new_team
                        current_score = new_score
                        team_names = {p.name for p in current_team}
                        
                        # Update candidate lists
                        front_candidates = [p for p in all_players if p.pos == "front" and p.name not in team_names]
                        back_candidates = [p for p in all_players if p.pos == "back" and p.name not in team_names]
                        
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        
        return current_team
    
    def _local_search_batch(
        self,
        candidate_teams: List[List[WildcardPlayerData]],
        all_players: List[WildcardPlayerData],
        max_price: float,
        w1: float,
        w2: float,
        w3: float,
        must_include: List[str] = None,
        top_k: int = 50
    ) -> List[List[WildcardPlayerData]]:
        """
        Apply local search to top candidates and return improved teams.
        Never removes must-include players.
        """
        if must_include is None:
            must_include = []
        
        if not candidate_teams:
            return []
        
        # Score all candidates first
        scored = []
        for team in candidate_teams:
            score = self._compute_team_weighted_score(team, w1, w2, w3)
            scored.append((score, team))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Apply local search to top k teams
        improved_teams = []
        seen_keys = set()
        
        print(f"   Applying local search to top {min(top_k, len(scored))} candidates...")
        
        for idx, (score, team) in enumerate(scored[:top_k]):
            improved = self._local_search_improve_team(
                team, all_players, max_price, w1, w2, w3, must_include
            )
            
            team_key = tuple(sorted(p.name for p in improved))
            if team_key not in seen_keys and len(improved) == 10:
                improved_teams.append(improved)
                seen_keys.add(team_key)
        
        # Also include remaining teams that weren't locally searched
        for score, team in scored[top_k:]:
            team_key = tuple(sorted(p.name for p in team))
            if team_key not in seen_keys:
                improved_teams.append(team)
                seen_keys.add(team_key)
        
        return improved_teams
    
    def _build_weekly_schedule_for_team(
        self,
        team: List[WildcardPlayerData],
        week_forms_getter,
        week_num: int
    ) -> Dict[int, List[Dict]]:
        """Build weekly schedule display data for a team."""
        schedule = {}
        
        for day in range(1, 8):
            day_players = []
            front_players = []
            back_players = []
            
            for p in team:
                forms = week_forms_getter(p)
                if forms[day - 1] > 0:
                    player_info = {
                        "Player": p.name,
                        "Form": round(forms[day - 1], 1),
                        "Pos": p.pos
                    }
                    if p.pos == "front":
                        front_players.append((forms[day - 1], player_info))
                    else:
                        back_players.append((forms[day - 1], player_info))
            
            # Sort by form descending
            front_players.sort(reverse=True, key=lambda x: x[0])
            back_players.sort(reverse=True, key=lambda x: x[0])
            
            # Get top 3 from each position
            top_front = [p[1] for p in front_players[:3]]
            top_back = [p[1] for p in back_players[:3]]
            
            # Combine and get top 5
            all_playing = [(p["Form"], p) for p in top_front + top_back]
            all_playing.sort(reverse=True, key=lambda x: x[0])
            top_5 = [p[1] for p in all_playing[:5]]
            
            day_players = top_5
            daily_total = sum(p["Form"] for p in day_players)
            
            schedule[day] = day_players + [{"Daily Total": round(daily_total, 1)}]
            
        return schedule
    
    def find_best_wildcard_teams(
        self,
        budget: float = 100.0,
        top_n: int = 5,
        must_include: List[str] = None,
        must_exclude: List[str] = None,
        injured: List[str] = None,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        candidate_pool_size: int = 500,
        processes: int = None
    ) -> List[Dict]:
        """
        Find the best wildcard team options.
        
        Args:
            budget: Total budget for team
            top_n: Number of top options to return
            must_include: Players that must be on the team
            must_exclude: Players that cannot be on the team
            injured: Injured players (excluded from consideration)
            w1: Weight for week 1 (default 0.5)
            w2: Weight for week 2 (default 0.3)
            w3: Weight for week 3 (default 0.2)
            candidate_pool_size: Number of candidates to generate
            processes: Number of processes for parallel computation
            
        Returns:
            List of dicts with team info, scores, schedules
        """
        if must_include is None:
            must_include = []
        if must_exclude is None:
            must_exclude = []
        if injured is None:
            injured = []
            
        # Normalize weights
        total_weight = w1 + w2 + w3
        if total_weight > 0:
            w1, w2, w3 = w1/total_weight, w2/total_weight, w3/total_weight
            
        print(f"🎯 Wildcard Optimizer starting...")
        print(f"   Budget: ${budget}, Weights: W1={w1:.2f}, W2={w2:.2f}, W3={w3:.2f}")
        print(f"   Must include: {must_include}")
        print(f"   Must exclude: {must_exclude}")
        print(f"   Injured: {injured}")
        
        t0 = time.perf_counter()
        
        # Build player data
        all_players = self._build_player_data(
            must_include, must_exclude, injured, w1, w2, w3
        )
        
        print(f"✅ Built data for {len(all_players)} eligible players")
        
        # Separate by position
        fronts = [p for p in all_players if p.pos == "front"]
        backs = [p for p in all_players if p.pos == "back"]
        
        print(f"   {len(fronts)} fronts, {len(backs)} backs")
        
        # Validate must-include players are in the pool
        for player_name in must_include:
            if player_name not in self._player_cache:
                print(f"⚠️ Must-include player '{player_name}' not found in players")
        
        # Generate candidate teams using diverse strategies
        print(f"🔄 Generating diverse candidate teams...")
        
        candidate_teams = self._generate_candidate_teams(
            fronts, backs, all_players, must_include, budget,
            w1=w1, w2=w2, w3=w3,
            num_perturbations=60,  # Increased from 30
            num_exclusion_runs=40,  # Increased from 20
            perturbation_strength=1.0  # Increased from 0.7
        )
        
        if not candidate_teams:
            print("❌ No valid teams found")
            return []
        
        # Apply local search to improve candidate teams
        print(f"🔍 Optimizing candidates with local search...")
        
        candidate_teams = self._local_search_batch(
            candidate_teams, all_players, budget, w1, w2, w3, 
            must_include=must_include,
            top_k=min(100, len(candidate_teams))  # Local search on top 100
        )
        
        print(f"   After local search: {len(candidate_teams)} unique teams")
        
        # Evaluate each candidate team
        print(f"📊 Evaluating teams...")
        
        evaluated = []
        must_include_set = set(must_include)
        
        for i, team in enumerate(candidate_teams):
            if (i + 1) % 100 == 0:
                print(f"   Evaluating team {i+1}/{len(candidate_teams)}...")
            
            # Verify must-include players are present
            team_player_names = {p.name for p in team}
            if not must_include_set.issubset(team_player_names):
                continue  # Skip teams missing must-include players
                
            # Compute accurate weekly scores
            week1_score, week2_score, week3_score = self._compute_team_weekly_scores(team)
            
            # Weighted total
            weighted_total = w1 * week1_score + w2 * week2_score + w3 * week3_score
            
            # Team price
            team_price = sum(p.price for p in team)
            
            evaluated.append({
                'team': team,
                'weighted_score': weighted_total,
                'week1_score': week1_score,
                'week2_score': week2_score,
                'week3_score': week3_score,
                'team_price': team_price
            })
        
        # Sort by weighted score descending
        evaluated.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Build results for top N
        results = []
        for item in evaluated[:top_n]:
            team = item['team']
            team_price = item['team_price']
            wallet_remaining = budget - team_price
            
            # Build team list for display
            team_list = [
                {
                    "Player": p.name,
                    "team": p.nba_team,
                    "Pos": p.pos,
                    "Salary": round(p.price, 1),
                    "MarketPrice": round(p.market_price, 1),
                    "Form": round(p.form, 1)
                }
                for p in team
            ]
            
            # Sort by position then form
            team_list.sort(key=lambda x: (0 if x["Pos"] == "front" else 1, -x["Form"]))
            
            # Build weekly schedules
            week1_schedule = self._build_weekly_schedule_for_team(
                team, lambda p: p.week1_forms, 1
            )
            week2_schedule = self._build_weekly_schedule_for_team(
                team, lambda p: p.week2_forms, 2
            )
            week3_schedule = self._build_weekly_schedule_for_team(
                team, lambda p: p.week3_forms, 3
            )
            
            results.append({
                'weighted_score': round(item['weighted_score'], 2),
                'week1_score': round(item['week1_score'], 1),
                'week2_score': round(item['week2_score'], 1),
                'week3_score': round(item['week3_score'], 1),
                'team_price': round(team_price, 1),
                'wallet_remaining': round(wallet_remaining, 1),
                'team': team_list,
                'week1_schedule': week1_schedule,
                'week2_schedule': week2_schedule,
                'week3_schedule': week3_schedule,
                'weights_used': {'w1': round(w1, 2), 'w2': round(w2, 2), 'w3': round(w3, 2)}
            })
        
        t1 = time.perf_counter()
        print(f"✅ Wildcard optimization complete in {t1-t0:.2f}s")
        print(f"   Found {len(results)} team options")
        
        return results


def test_wildcard_optimizer():
    """Test function for wildcard optimizer."""
    # This would be used for testing
    print("Wildcard optimizer module loaded successfully")


if __name__ == "__main__":
    test_wildcard_optimizer()
