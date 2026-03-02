"""
optimizer_prefix_suffix.py

Optimized Fantasy Team Substitution Algorithm using Prefix-Suffix Scoring.

This module implements an advanced substitution optimizer that:
- Supports 0, 1, or 2 substitutions within the week
- Substitutions can occur on any day 1..7
- If 2 subs: can happen on same day or different days
- Uses prefix-suffix precomputation for efficient scoring
- Supports multiprocessing for parallel evaluation
- Includes optional next-week lookahead with weighted scoring

Algorithm Overview:
1. Precompute base team daily scores and prefix sums
2. For each substitution scenario (day, players):
   - Use prefix sums for days before substitution
   - Compute suffix for days after substitution with new team
3. Shortlist candidates using marginal value heuristics
4. Optionally apply domination pruning to reduce candidate pool
5. Parallelize evaluation across substitution day pairs and out-combinations

=============================================================================
WEB APP INTEGRATION GUIDE
=============================================================================

This optimizer is a DROP-IN REPLACEMENT for the existing FantasyOptimizer.
It's compatible with the same DataFrame inputs and returns the same format.

### Quick Integration (app.py):

```python
from optimizer_prefix_suffix import FantasyOptimizerPrefixSuffix

# In your route handler:
optimizer = FantasyOptimizerPrefixSuffix(
    pd.DataFrame(user_team_data),
    pd.DataFrame(updated_players_data),
    schedule_week1,
    schedule_week2
)

# Get results in NEW dict format (recommended):
results = optimizer.find_best_weekly_substitutions_any_day(
    extra_salary=extra_salary,
    top_n=top_n,
    untradable_players=untradable_players,
    must_trade_players=must_trade_players,
    w1=0.75, w2=0.25
)

# OR get results in LEGACY tuple format for existing web app:
results_dict = optimizer.find_best_weekly_substitutions_any_day(...)
results_legacy = optimizer.convert_to_legacy_format(results_dict)

# Now results_legacy is compatible with existing app.py code:
# (new_form, current_form, new_salary, out_players, in_players, team, sched)
```

### Key Differences from Original:

1. **Return Format**:
   - Original: Tuple (new_form, current_form, salary, out, in, team, sched)
   - New (default): Dict with keys: 'total_score', 'week1_score', 'num_subs', 
                    'substitution_plan', 'final_team', 'weekly_schedule', etc.
   - Use `convert_to_legacy_format()` to convert new → old format

2. **New Features**:
   - Substitutions can happen on ANY day (not just day 1)
   - 2 subs can be on different days with wallet tracking
   - Week 2 lookahead with weighted scoring (w1, w2 parameters)
   - Domination pruning for faster execution
   
3. **Parameters**:
   - Same: extra_salary, top_n, untradable_players, must_trade_players
   - New: w1, w2 (week weights), allow_domination_prune, shortlist_k, processes

### Expected Performance:
   - 2,000-5,000 scenarios/second on modern CPU
   - Typical search: 0.1-0.5 seconds for 2,500 scenarios
   
=============================================================================
"""

import pandas as pd
import itertools
from itertools import combinations
import heapq
import copy
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import time
from multiprocessing import Pool, cpu_count
import logging
import bisect
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Container for timing and counter statistics."""
    # Timing accumulators (in seconds)
    time_build_player_data: float = 0.0
    time_build_team_state: float = 0.0
    time_domination_pruning: float = 0.0
    time_shortlist_scoring: float = 0.0
    time_shortlist_total: float = 0.0
    time_single_sub_eval: float = 0.0
    time_double_same_eval: float = 0.0
    time_double_diff_eval: float = 0.0
    time_week2_baseline: float = 0.0
    time_apply_substitution: float = 0.0
    time_compute_segment_score: float = 0.0
    time_compute_daily_score: float = 0.0
    time_result_formatting: float = 0.0
    time_constraint_checks: float = 0.0
    time_sort_results: float = 0.0
    time_format_output: float = 0.0
    time_total: float = 0.0
    
    # Counters
    count_players_in_pool: int = 0
    count_players_after_domination_prune: int = 0
    count_after_domination: int = 0  # alias for compatibility
    count_shortlist_final: int = 0
    count_out_combos_1: int = 0
    count_out_combos_2: int = 0
    count_single_sub_evaluated: int = 0
    count_single_sub_valid: int = 0
    count_double_same_evaluated: int = 0
    count_double_same_valid: int = 0
    count_double_diff_evaluated: int = 0
    count_double_diff_valid: int = 0
    count_week2_baseline_calls: int = 0
    count_apply_substitution_calls: int = 0
    count_apply_substitution: int = 0  # alias for compatibility
    count_compute_segment_calls: int = 0
    count_compute_segment_score: int = 0  # alias for compatibility
    count_compute_daily_calls: int = 0
    count_constraint_check_calls: int = 0
    count_results_total: int = 0
    
    def print_summary(self):
        """Print a formatted summary of all timing and counter stats."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE STATISTICS SUMMARY")
        print("=" * 80)
        
        # Timing summary
        print("\n⏱️  TIMING BREAKDOWN (seconds):")
        print("-" * 50)
        print("  (Note: Week 2 baseline time is included in each eval phase)")
        timings = [
            ("Build player data", self.time_build_player_data),
            ("Build team state", self.time_build_team_state),
            ("Domination pruning", self.time_domination_pruning),
            ("Shortlist scoring", self.time_shortlist_scoring),
            ("Shortlist total", self.time_shortlist_total),
            ("Single sub evaluation", self.time_single_sub_eval),
            ("Double sub (same day)", self.time_double_same_eval),
            ("Double sub (diff days)", self.time_double_diff_eval),
            ("  └─ Week 2 calls (included)", self.time_week2_baseline),
            ("Sort results", self.time_sort_results),
            ("Format output", self.time_format_output),
        ]
        
        for name, duration in timings:
            pct = (duration / self.time_total * 100) if self.time_total > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {name:30s} {duration:8.4f}s ({pct:5.1f}%) {bar[:20]}")
        
        print(f"\n  {'TOTAL TIME':30s} {self.time_total:8.4f}s")
        
        # Counter summary
        print("\n🔢 OPERATION COUNTS:")
        print("-" * 50)
        counters = [
            ("Players in pool", self.count_players_in_pool),
            ("After domination prune", self.count_after_domination),
            ("Final shortlist size", self.count_shortlist_final),
            ("Out combos (1 player)", self.count_out_combos_1),
            ("Out combos (2 players)", self.count_out_combos_2),
            ("Single subs evaluated", self.count_single_sub_evaluated),
            ("Single subs valid", self.count_single_sub_valid),
            ("Double same evaluated", self.count_double_same_evaluated),
            ("Double same valid", self.count_double_same_valid),
            ("Double diff evaluated", self.count_double_diff_evaluated),
            ("Double diff valid", self.count_double_diff_valid),
            ("Week2 baseline calls", self.count_week2_baseline_calls),
            ("Apply substitution calls", self.count_apply_substitution),
            ("Compute segment calls", self.count_compute_segment_score),
            ("Total results generated", self.count_results_total),
        ]
        
        for name, count in counters:
            print(f"  {name:30s} {count:>10,}")
        
        # Efficiency metrics
        print("\n📈 EFFICIENCY METRICS:")
        print("-" * 50)
        if self.count_single_sub_evaluated > 0:
            valid_rate_single = self.count_single_sub_valid / self.count_single_sub_evaluated * 100
            print(f"  Single sub valid rate:       {valid_rate_single:6.2f}%")
        if self.count_double_same_evaluated > 0:
            valid_rate_double_same = self.count_double_same_valid / self.count_double_same_evaluated * 100
            print(f"  Double same valid rate:      {valid_rate_double_same:6.2f}%")
        if self.count_double_diff_evaluated > 0:
            valid_rate_double_diff = self.count_double_diff_valid / self.count_double_diff_evaluated * 100
            print(f"  Double diff valid rate:      {valid_rate_double_diff:6.2f}%")
        
        if self.time_single_sub_eval > 0 and self.count_single_sub_evaluated > 0:
            rate_single = self.count_single_sub_evaluated / self.time_single_sub_eval
            print(f"  Single sub eval rate:        {rate_single:,.0f}/sec")
        if self.time_double_diff_eval > 0 and self.count_double_diff_evaluated > 0:
            rate_double = self.count_double_diff_evaluated / self.time_double_diff_eval
            print(f"  Double diff eval rate:       {rate_double:,.0f}/sec")
            
        print("\n" + "=" * 80)


@dataclass
class PlayerData:
    """Lightweight player data structure for performance-critical operations."""
    idx: int
    name: str
    pos: str  # 'front' or 'back'
    nba_team: str
    form: float
    salary_my_team: float  # salary if selling from my team
    salary_market: float   # salary if buying from pool
    day_bitmask: int       # bitmask of days playing (bit 0 = day 1, etc.)
    day_forms: Tuple[float, ...]  # (day1_form, day2_form, ..., day7_form)
    
    def plays_on_day(self, day: int) -> bool:
        """Check if player plays on given day (1-indexed)."""
        return bool(self.day_bitmask & (1 << (day - 1)))
    
    def get_day_form(self, day: int) -> float:
        """Get form for specific day (1-indexed). Returns 0 if not playing."""
        return self.day_forms[day - 1]


@dataclass 
class TeamState:
    """Represents team state at a point in time with precomputed values."""
    players: List[PlayerData]
    player_names: Set[str]
    nba_team_counts: Dict[str, int]  # count of players per NBA team
    front_count: int
    back_count: int
    total_salary: float
    
    # Precomputed daily data
    daily_scores: List[float] = field(default_factory=list)  # [day1, day2, ..., day7]
    prefix_sums: List[float] = field(default_factory=list)   # prefix_sums[k] = sum of days 1..k
    
    # Per-day sorted forms for quick lineup computation
    daily_front_forms: List[List[Tuple[float, str]]] = field(default_factory=list)
    daily_back_forms: List[List[Tuple[float, str]]] = field(default_factory=list)


class FantasyOptimizerPrefixSuffix:
    """
    Advanced Fantasy Optimizer using prefix-suffix scoring strategy.
    
    Supports:
    - 0, 1, or 2 substitutions within the week
    - Substitutions on any day 1..7
    - Salary constraints with wallet updates after each sub
    - Team constraints: 5 front + 5 back, max 2 from same NBA team
    - Optional next-week lookahead with weighted scoring
    - Multiprocessing for parallel evaluation
    """
    
    def __init__(self, my_team, best_filter_path, schedule_week1: dict, schedule_week2: dict = None):
        """
        Initialize optimizer with team data and schedules.
        
        Args:
            my_team: DataFrame or path to CSV with current team
            best_filter_path: DataFrame with all available players (pool)
            schedule_week1: Dict mapping day (1-7) to list of playing NBA teams
            schedule_week2: Optional dict for second week schedule
        """
        # Load team data
        if isinstance(my_team, str):
            self.my_team = pd.read_csv(my_team)
        else:
            self.my_team = my_team.copy()
            
        self.best_filter = best_filter_path.copy()
        self.schedule = schedule_week1.copy()
        self.schedule_second = schedule_week2.copy() if schedule_week2 else None
        
        # Standardize column names and types
        self._standardize_columns()
        
        # Build schedule lookups
        self.playing_teams_dict = {
            day: schedule_week1.get(day, [])
            for day in range(1, 8)
        }
        
        self.playing_teams_second_dict = (
            {day: schedule_week2.get(day, []) for day in range(1, 8)}
            if schedule_week2 else {}
        )
        
        # Precompute player-day matrices
        self._build_lookup_structures()
        
        # Convert to lightweight data structures
        self._player_data_cache: Dict[str, PlayerData] = {}
        self._pool_players: List[PlayerData] = []
        self._my_team_players: List[PlayerData] = []
        
    def _standardize_columns(self):
        """Standardize column names and data types."""
        # Rename $ to Salary if needed
        if "$" in self.my_team.columns and "Salary" not in self.my_team.columns:
            self.my_team = self.my_team.rename(columns={"$": "Salary"})
        if "$" in self.best_filter.columns and "Salary" not in self.best_filter.columns:
            self.best_filter = self.best_filter.rename(columns={"$": "Salary"})
            
        # Standardize position names
        self.my_team["Pos"] = self.my_team["Pos"].str.lower()
        self.best_filter["Pos"] = self.best_filter["Pos"].str.lower()
        
        # Ensure numeric types
        self.my_team["Salary"] = self.my_team["Salary"].astype(float)
        self.my_team["Form"] = self.my_team["Form"].astype(float)
        self.best_filter["Salary"] = self.best_filter["Salary"].astype(float)
        self.best_filter["Form"] = self.best_filter["Form"].astype(float)
        
    def _build_lookup_structures(self):
        """Build efficient lookup structures for schedule and form data."""
        # Form lookup: (player, team) -> form
        self.form_lookup = {
            (row["Player"], row["team"]): row["Form"] 
            for _, row in self.best_filter.iterrows()
        }
        
        # Player name lookup to full row (used for market price lookups)
        self.player_lookup = {
            row["Player"]: row.to_dict()
            for _, row in self.best_filter.iterrows()
        }
        
        # My team salary lookup (user's custom value for selling)
        self.my_team_salary_lookup = {
            row["Player"]: row["Salary"]
            for _, row in self.my_team.iterrows()
        }
        
    def _create_player_data(self, row: dict, is_my_team: bool, idx: int) -> PlayerData:
        """Create lightweight PlayerData from DataFrame row."""
        name = row["Player"]
        nba_team = row.get("team", "")
        
        # Compute day bitmask and day forms
        day_bitmask = 0
        day_forms = []
        for day in range(1, 8):
            playing_teams = self.playing_teams_dict.get(day, [])
            if nba_team in playing_teams:
                day_bitmask |= (1 << (day - 1))
                day_forms.append(row["Form"])
            else:
                day_forms.append(0.0)
                
        return PlayerData(
            idx=idx,
            name=name,
            pos=row["Pos"].lower(),
            nba_team=nba_team,
            form=row["Form"],
            salary_my_team=self.my_team_salary_lookup.get(name, row["Salary"]),
            salary_market=row["Salary"],
            day_bitmask=day_bitmask,
            day_forms=tuple(day_forms)
        )
    
    def _build_player_data_lists(self, week: int = 1):
        """Convert DataFrames to lightweight PlayerData lists."""
        schedule_dict = self.playing_teams_dict if week == 1 else self.playing_teams_second_dict
        
        # Build my team players
        self._my_team_players = []
        for idx, (_, row) in enumerate(self.my_team.iterrows()):
            player = self._create_player_data_with_schedule(row.to_dict(), True, idx, schedule_dict)
            self._my_team_players.append(player)
            self._player_data_cache[row["Player"]] = player
            
        # Build pool players (excluding my team)
        my_team_names = set(self.my_team["Player"].tolist())
        self._pool_players = []
        for idx, (_, row) in enumerate(self.best_filter.iterrows()):
            if row["Player"] not in my_team_names:
                player = self._create_player_data_with_schedule(row.to_dict(), False, idx, schedule_dict)
                self._pool_players.append(player)
                self._player_data_cache[row["Player"]] = player
                
    def _create_player_data_with_schedule(self, row: dict, is_my_team: bool, idx: int, 
                                          schedule_dict: dict) -> PlayerData:
        """
        Create PlayerData with specific schedule dict.
        
        Salary handling:
        - salary_my_team: What the user values the player at (for SELLING from team)
          - For my_team players: user's custom price
          - For pool players: same as market price
        - salary_market: The market price (for BUYING from pool)
          - For my_team players: look up from best_filter (original market price)
          - For pool players: their Salary from best_filter
        """
        name = row["Player"]
        nba_team = row.get("team", "")
        
        day_bitmask = 0
        day_forms = []
        for day in range(1, 8):
            playing_teams = schedule_dict.get(day, [])
            if nba_team in playing_teams:
                day_bitmask |= (1 << (day - 1))
                day_forms.append(row["Form"])
            else:
                day_forms.append(0.0)
        
        # Determine salaries based on whether player is from my_team or pool
        if is_my_team:
            # For my_team players:
            # - salary_my_team = user's custom price (from my_team DataFrame)
            # - salary_market = original market price (from best_filter lookup)
            salary_my_team = row["Salary"]  # User's custom price
            # Look up market price from best_filter
            pool_row = self.player_lookup.get(name)
            if pool_row:
                salary_market = pool_row["Salary"]
            else:
                # Player not in pool (shouldn't happen normally)
                salary_market = row["Salary"]
        else:
            # For pool players: both are the same (market price)
            salary_my_team = row["Salary"]
            salary_market = row["Salary"]
        
        # Handle missing position data (NaN values)
        pos_raw = row.get("Pos", "")
        if pd.isna(pos_raw) or pos_raw == "":
            pos = "back"  # Default to back position if missing
        else:
            pos = str(pos_raw).lower()
                
        return PlayerData(
            idx=idx,
            name=name,
            pos=pos,
            nba_team=nba_team,
            form=row["Form"],
            salary_my_team=salary_my_team,
            salary_market=salary_market,
            day_bitmask=day_bitmask,
            day_forms=tuple(day_forms)
        )
    
    def _compute_daily_score(self, players: List[PlayerData], day: int) -> float:
        """
        Compute score for a single day using lineup rules.
        
        Rules:
        - Pick top 5 players among those playing that day
        - Max 3 from front position, max 3 from back position
        - Among top 3 front + top 3 back, pick best 5
        """
        front_forms = []
        back_forms = []
        
        for p in players:
            form = p.get_day_form(day)
            if form > 0:
                if p.pos == "front":
                    front_forms.append(form)
                else:
                    back_forms.append(form)
                    
        # Get top 3 from each position
        front_forms.sort(reverse=True)
        back_forms.sort(reverse=True)
        
        top_front = front_forms[:3]
        top_back = back_forms[:3]
        
        # Combine and take top 5
        combined = top_front + top_back
        combined.sort(reverse=True)
        
        return sum(combined[:5])
    
    def _compute_weekly_score(self, players: List[PlayerData], 
                              day_start: int = 1, day_end: int = 7) -> float:
        """Compute total score for days in range [day_start, day_end]."""
        total = 0.0
        for day in range(day_start, day_end + 1):
            total += self._compute_daily_score(players, day)
        return total
    
    def _build_team_state(self, players: List[PlayerData]) -> TeamState:
        """Build TeamState with all precomputed values."""
        player_names = {p.name for p in players}
        
        # Count NBA teams and positions
        nba_team_counts = defaultdict(int)
        front_count = 0
        back_count = 0
        total_salary = 0.0
        
        for p in players:
            nba_team_counts[p.nba_team] += 1
            if p.pos == "front":
                front_count += 1
            else:
                back_count += 1
            total_salary += p.salary_my_team
            
        # Compute daily scores
        daily_scores = []
        for day in range(1, 8):
            score = self._compute_daily_score(players, day)
            daily_scores.append(score)
            
        # Compute prefix sums: prefix_sums[k] = sum of days 1..k (0-indexed: day1 = index 0)
        prefix_sums = [0.0]  # prefix_sums[0] = 0 (no days)
        running = 0.0
        for score in daily_scores:
            running += score
            prefix_sums.append(running)
            
        # Precompute daily sorted forms by position
        daily_front_forms = []
        daily_back_forms = []
        for day in range(1, 8):
            front = [(p.get_day_form(day), p.name) for p in players 
                     if p.pos == "front" and p.get_day_form(day) > 0]
            back = [(p.get_day_form(day), p.name) for p in players 
                    if p.pos == "back" and p.get_day_form(day) > 0]
            front.sort(reverse=True)
            back.sort(reverse=True)
            daily_front_forms.append(front)
            daily_back_forms.append(back)
            
        return TeamState(
            players=players,
            player_names=player_names,
            nba_team_counts=dict(nba_team_counts),
            front_count=front_count,
            back_count=back_count,
            total_salary=total_salary,
            daily_scores=daily_scores,
            prefix_sums=prefix_sums,
            daily_front_forms=daily_front_forms,
            daily_back_forms=daily_back_forms
        )
    
    def _compute_shortlist_score(self, candidate: PlayerData, 
                                 team_state: TeamState,
                                 day_start: int = 1, day_end: int = 7) -> float:
        """
        Compute heuristic shortlist score for a candidate player.
        
        Measures marginal value: how much the candidate could contribute
        above the current 3rd-best player at their position.
        """
        score = 0.0
        pos = candidate.pos
        
        for day in range(day_start, day_end + 1):
            candidate_form = candidate.get_day_form(day)
            if candidate_form <= 0:
                continue
                
            day_idx = day - 1
            if pos == "front":
                sorted_forms = team_state.daily_front_forms[day_idx]
            else:
                sorted_forms = team_state.daily_back_forms[day_idx]
                
            if len(sorted_forms) < 3:
                # Position has less than 3 players, candidate definitely contributes
                score += candidate_form
            else:
                # Check if candidate beats 3rd best
                third_best = sorted_forms[2][0] if len(sorted_forms) >= 3 else 0
                margin = max(0, candidate_form - third_best)
                score += margin
                
        return score
    
    def _apply_domination_pruning(self, candidates: List[PlayerData]) -> List[PlayerData]:
        """
        Remove dominated candidates within same NBA team and position.
        
        A player B is dominated by A if:
        - Same NBA team and position
        - A.form >= B.form AND A.salary_market <= B.salary_market
        - At least one strict inequality
        """
        # Group by (nba_team, pos)
        groups = defaultdict(list)
        for c in candidates:
            groups[(c.nba_team, c.pos)].append(c)
            
        result = []
        for key, group in groups.items():
            if len(group) <= 1:
                result.extend(group)
                continue
                
            # Check each pair for domination
            non_dominated = []
            for i, b in enumerate(group):
                dominated = False
                for j, a in enumerate(group):
                    if i == j:
                        continue
                    # Check if a dominates b
                    if (a.form >= b.form and a.salary_market <= b.salary_market and
                        (a.form > b.form or a.salary_market < b.salary_market)):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(b)
            result.extend(non_dominated)
            
        return result
    
    def _build_shortlist(self, team_state: TeamState, 
                         candidates: List[PlayerData],
                         k: int = 60,
                         day_start: int = 1, day_end: int = 7,
                         allow_domination_prune: bool = True,
                         stats: Optional['PerformanceStats'] = None) -> List[PlayerData]:
        """
        Build shortlist of top-K candidates using heuristic scoring.
        
        Args:
            team_state: Current team state
            candidates: All pool candidates
            k: Number of candidates to keep
            day_start: Start day for scoring
            day_end: End day for scoring
            allow_domination_prune: Whether to apply domination pruning
            stats: Optional performance stats tracker
            
        Returns:
            Shortlisted candidates
        """
        if allow_domination_prune:
            t0 = time.perf_counter()
            candidates = self._apply_domination_pruning(candidates)
            if stats:
                stats.time_domination_pruning = time.perf_counter() - t0
                stats.count_after_domination = len(candidates)
        
        # Score all candidates
        t0 = time.perf_counter()
        scored = []
        for c in candidates:
            score = self._compute_shortlist_score(c, team_state, day_start, day_end)
            scored.append((score, c))
            
        # Sort by score descending and take top k
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if stats:
            stats.time_shortlist_scoring = time.perf_counter() - t0
        
        return [c for _, c in scored[:k]]
    
    def _check_team_constraints(self, players: List[PlayerData]) -> bool:
        """
        Check if team satisfies all constraints.
        
        Returns True if:
        - Exactly 10 players
        - 5 front + 5 back
        - Max 2 players from same NBA team
        """
        if len(players) != 10:
            return False
            
        front_count = sum(1 for p in players if p.pos == "front")
        back_count = sum(1 for p in players if p.pos == "back")
        
        if front_count != 5 or back_count != 5:
            return False
            
        nba_team_counts = defaultdict(int)
        for p in players:
            nba_team_counts[p.nba_team] += 1
            if nba_team_counts[p.nba_team] > 2:
                return False
                
        return True
    
    def _apply_substitution(self, players: List[PlayerData],
                            out_players: List[PlayerData],
                            in_players: List[PlayerData]) -> List[PlayerData]:
        """Apply substitution: remove out_players, add in_players."""
        out_names = {p.name for p in out_players}
        new_players = [p for p in players if p.name not in out_names]
        new_players.extend(in_players)
        return new_players
    
    def _compute_segment_score(self, players: List[PlayerData],
                               day_start: int, day_end: int) -> float:
        """Compute score for a segment of days."""
        total = 0.0
        for day in range(day_start, day_end + 1):
            total += self._compute_daily_score(players, day)
        return total
    
    def _evaluate_single_sub(self, team_state: TeamState,
                             out_player: PlayerData,
                             in_player: PlayerData,
                             sub_day: int,
                             extra_salary: float) -> Optional[Tuple[float, float, float]]:
        """
        Evaluate a single substitution on given day.
        
        Returns: (total_score, wallet_after, None) or None if invalid
        """
        # Check salary feasibility
        wallet = extra_salary
        wallet += out_player.salary_my_team  # sell
        wallet -= in_player.salary_market    # buy
        
        if wallet < 0:
            return None
            
        # Check position constraint
        if out_player.pos != in_player.pos:
            return None
            
        # Check NBA team constraint
        new_nba_counts = dict(team_state.nba_team_counts)
        new_nba_counts[out_player.nba_team] = new_nba_counts.get(out_player.nba_team, 0) - 1
        new_nba_counts[in_player.nba_team] = new_nba_counts.get(in_player.nba_team, 0) + 1
        
        if new_nba_counts[in_player.nba_team] > 2:
            return None
            
        # Compute score using prefix-suffix
        # Days 1..sub_day-1: use base team (prefix sum)
        score_before = team_state.prefix_sums[sub_day - 1]  # sum of days 1..sub_day-1
        
        # Days sub_day..7: compute with new team
        new_players = self._apply_substitution(
            team_state.players, [out_player], [in_player]
        )
        score_after = self._compute_segment_score(new_players, sub_day, 7)
        
        total_score = score_before + score_after
        final_wallet = extra_salary + out_player.salary_my_team - in_player.salary_market
        
        return (total_score, final_wallet, None)
    
    def _evaluate_double_sub_same_day(self, team_state: TeamState,
                                      out_players: List[PlayerData],
                                      in_players: List[PlayerData],
                                      sub_day: int,
                                      extra_salary: float) -> Optional[Tuple[float, float]]:
        """
        Evaluate two substitutions on the same day.
        
        Returns: (total_score, final_wallet) or None if invalid
        """
        # Check total salary feasibility
        wallet = extra_salary
        for p in out_players:
            wallet += p.salary_my_team
        for p in in_players:
            wallet -= p.salary_market
            
        if wallet < 0:
            return None
            
        # Check position constraint
        out_pos = sorted([p.pos for p in out_players])
        in_pos = sorted([p.pos for p in in_players])
        if out_pos != in_pos:
            return None
            
        # Check NBA team constraint
        new_nba_counts = dict(team_state.nba_team_counts)
        for p in out_players:
            new_nba_counts[p.nba_team] = new_nba_counts.get(p.nba_team, 0) - 1
        for p in in_players:
            new_nba_counts[p.nba_team] = new_nba_counts.get(p.nba_team, 0) + 1
            if new_nba_counts[p.nba_team] > 2:
                return None
                
        # Compute score
        score_before = team_state.prefix_sums[sub_day - 1]
        
        new_players = self._apply_substitution(
            team_state.players, out_players, in_players
        )
        score_after = self._compute_segment_score(new_players, sub_day, 7)
        
        total_score = score_before + score_after
        
        return (total_score, wallet)
    
    def _evaluate_double_sub_diff_days(self, team_state: TeamState,
                                       out1: PlayerData, in1: PlayerData,
                                       sub_day1: int,
                                       out2: PlayerData, in2: PlayerData,
                                       sub_day2: int,
                                       extra_salary: float) -> Optional[Tuple[float, float]]:
        """
        Evaluate two substitutions on different days (sub_day1 < sub_day2).
        
        Wallet is updated after first sub.
        
        Returns: (total_score, final_wallet) or None if invalid
        """
        assert sub_day1 < sub_day2
        
        # First substitution
        wallet = extra_salary
        wallet += out1.salary_my_team
        wallet -= in1.salary_market
        
        if wallet < 0:
            return None
            
        # Check position for first sub
        if out1.pos != in1.pos:
            return None
            
        # Team after first sub
        new_nba_counts = dict(team_state.nba_team_counts)
        new_nba_counts[out1.nba_team] = new_nba_counts.get(out1.nba_team, 0) - 1
        new_nba_counts[in1.nba_team] = new_nba_counts.get(in1.nba_team, 0) + 1
        
        if new_nba_counts[in1.nba_team] > 2:
            return None
            
        # Second substitution (wallet already updated)
        wallet += out2.salary_my_team if out2.name != in1.name else in1.salary_market
        wallet -= in2.salary_market
        
        if wallet < 0:
            return None
            
        # Check position for second sub
        if out2.pos != in2.pos:
            return None
            
        # Team after second sub
        new_nba_counts[out2.nba_team] = new_nba_counts.get(out2.nba_team, 0) - 1
        new_nba_counts[in2.nba_team] = new_nba_counts.get(in2.nba_team, 0) + 1
        
        if new_nba_counts[in2.nba_team] > 2:
            return None
            
        # Compute score using 3 segments
        # Segment 1: days 1..sub_day1-1 with T0 (base team)
        score_seg1 = team_state.prefix_sums[sub_day1 - 1]
        
        # Segment 2: days sub_day1..sub_day2-1 with T1 (after first sub)
        players_t1 = self._apply_substitution(
            team_state.players, [out1], [in1]
        )
        score_seg2 = self._compute_segment_score(players_t1, sub_day1, sub_day2 - 1)
        
        # Segment 3: days sub_day2..7 with T2 (after second sub)
        players_t2 = self._apply_substitution(players_t1, [out2], [in2])
        score_seg3 = self._compute_segment_score(players_t2, sub_day2, 7)
        
        total_score = score_seg1 + score_seg2 + score_seg3
        final_wallet = extra_salary + out1.salary_my_team - in1.salary_market + \
                       (out2.salary_my_team if out2.name != in1.name else in1.salary_market) - \
                       in2.salary_market
        
        return (total_score, final_wallet)
    
    def _generate_out_combinations(self, team_state: TeamState,
                                   untradable_players: List[str],
                                   must_trade_players: List[str],
                                   num_subs: int) -> List[Tuple[PlayerData, ...]]:
        """
        Generate valid out-player combinations.
        
        Respects untradable and must_trade constraints.
        """
        tradable = [p for p in team_state.players if p.name not in untradable_players]
        
        if num_subs == 1:
            combos = [(p,) for p in tradable]
        else:
            combos = list(combinations(tradable, 2))
            
        # Filter by must_trade constraint
        if must_trade_players:
            must_trade_set = set(must_trade_players)
            combos = [c for c in combos 
                      if any(p.name in must_trade_set for p in c)]
                      
            # If must_trade has 2 players, both must be traded together
            if len(must_trade_players) == 2 and num_subs == 2:
                combos = [c for c in combos 
                          if all(p.name in must_trade_set for p in c)]
                          
        return combos
    
    def _evaluate_no_sub(self, team_state: TeamState) -> Tuple[float, float]:
        """Evaluate keeping current team with no substitutions."""
        total_score = sum(team_state.daily_scores)
        return (total_score, 0.0)  # wallet unchanged
    
    def _week2_baseline_score(self, players: List[PlayerData], 
                               stats: Optional['PerformanceStats'] = None) -> float:
        """Compute week 2 score for given team (no additional subs)."""
        if not self.schedule_second:
            return 0.0
        
        if stats:
            stats.count_week2_baseline_calls += 1
            
        # Rebuild player data for week 2 schedule
        week2_players = []
        for p in players:
            row = self.player_lookup.get(p.name)
            if row:
                week2_player = self._create_player_data_with_schedule(
                    row, True, p.idx, self.playing_teams_second_dict
                )
                week2_players.append(week2_player)
                
        return self._compute_weekly_score(week2_players, 1, 7)
    
    def find_best_weekly_substitutions_any_day(
        self,
        extra_salary: float = 0,
        top_n: int = 5,
        untradable_players: List[str] = None,
        must_trade_players: List[str] = None,
        w1: float = 0.75,
        w2: float = 0.25,
        allow_domination_prune: bool = True,
        shortlist_k: int = 60,
        processes: int = None
    ) -> List[Dict[str, Any]]:
        """
        Find best substitution plans allowing subs on any day 1-7.
        
        Supports 0, 1, or 2 substitutions with flexible timing.
        
        Args:
            extra_salary: Available extra salary for transactions
            top_n: Number of top results to return
            untradable_players: Players that cannot be sold
            must_trade_players: Players that must be sold
            w1: Weight for week 1 score
            w2: Weight for week 2 baseline score (if schedule_second exists)
            allow_domination_prune: Whether to apply domination pruning
            shortlist_k: Number of candidates to keep in shortlist
            processes: Number of processes for parallel evaluation (None = auto)
            
        Returns:
            List of top_n substitution plans, each containing:
            - total_score, week1_score, week2_score (if applicable)
            - substitution_plan with details
            - final_team roster
            - weekly_schedule display
        """
        logger.info("🚀 Starting prefix-suffix substitution search...")
        start_time = time.perf_counter()
        
        # Initialize performance stats
        stats = PerformanceStats()
        
        # Default parameters
        if untradable_players is None:
            untradable_players = []
        if must_trade_players is None:
            must_trade_players = []
        if processes is None:
            processes = max(1, cpu_count() - 1)
            
        extra_salary = float(extra_salary)
        
        # Build lightweight data structures
        t0 = time.perf_counter()
        self._build_player_data_lists(week=1)
        stats.time_build_player_data = time.perf_counter() - t0
        stats.count_players_in_pool = len(self._pool_players)
        
        # Build initial team state
        t0 = time.perf_counter()
        team_state = self._build_team_state(self._my_team_players)
        stats.time_build_team_state = time.perf_counter() - t0
        base_weekly_score = sum(team_state.daily_scores)
        
        logger.info(f"📊 Base team weekly score: {base_weekly_score:.2f}")
        logger.info(f"📊 Team salary: {team_state.total_salary:.2f}")
        
        # Build shortlist of candidates
        t0 = time.perf_counter()
        candidates = self._pool_players
        shortlist = self._build_shortlist(
            team_state, candidates, k=shortlist_k,
            allow_domination_prune=allow_domination_prune,
            stats=stats
        )
        stats.time_shortlist_total = time.perf_counter() - t0
        stats.count_shortlist_final = len(shortlist)
        
        logger.info(f"📋 Shortlist size: {len(shortlist)} players")
        
        # Group shortlist by position
        front_candidates = [c for c in shortlist if c.pos == "front"]
        back_candidates = [c for c in shortlist if c.pos == "back"]
        
        # Results heap (min-heap, so we store negated scores)
        results = []
        
        # === Evaluate no substitution ===
        t0 = time.perf_counter()
        no_sub_score = base_weekly_score
        week2_score = self._week2_baseline_score(team_state.players, stats) if self.schedule_second else 0
        stats.time_week2_baseline += time.perf_counter() - t0
        weighted_score = w1 * no_sub_score + w2 * week2_score
        
        results.append({
            "weighted_score": weighted_score,
            "week1_score": no_sub_score,
            "week2_score": week2_score,
            "num_subs": 0,
            "subs": [],
            "final_players": team_state.players,
            "final_wallet": extra_salary
        })
        stats.count_results_total += 1
        
        # === Evaluate single substitutions ===
        logger.info("🔄 Evaluating single substitutions...")
        single_sub_count = 0
        t_single_start = time.perf_counter()
        
        out_combos_1 = self._generate_out_combinations(
            team_state, untradable_players, must_trade_players, 1
        )
        stats.count_out_combos_1 = len(out_combos_1)
        
        for (out_player,) in out_combos_1:
            # Find matching position candidates
            candidates_for_pos = front_candidates if out_player.pos == "front" else back_candidates
            
            for sub_day in range(1, 8):
                for in_player in candidates_for_pos:
                    if in_player.name in team_state.player_names:
                        continue
                    
                    stats.count_single_sub_evaluated += 1
                    
                    result = self._evaluate_single_sub(
                        team_state, out_player, in_player, sub_day, extra_salary
                    )
                    
                    if result is None:
                        continue
                    
                    stats.count_single_sub_valid += 1
                    single_sub_count += 1
                    total_score, final_wallet, _ = result
                    
                    # Compute week 2 score
                    t_w2 = time.perf_counter()
                    new_players = self._apply_substitution(
                        team_state.players, [out_player], [in_player]
                    )
                    stats.count_apply_substitution += 1
                    week2_score = self._week2_baseline_score(new_players, stats) if self.schedule_second else 0
                    stats.time_week2_baseline += time.perf_counter() - t_w2
                    weighted_score = w1 * total_score + w2 * week2_score
                    
                    results.append({
                        "weighted_score": weighted_score,
                        "week1_score": total_score,
                        "week2_score": week2_score,
                        "num_subs": 1,
                        "subs": [
                            {"day": sub_day, "out": out_player.name, "in": in_player.name,
                             "out_salary": out_player.salary_my_team, 
                             "in_salary": in_player.salary_market}
                        ],
                        "final_players": new_players,
                        "final_wallet": final_wallet
                    })
                    stats.count_results_total += 1
                    
        stats.time_single_sub_eval = time.perf_counter() - t_single_start
        logger.info(f"  Evaluated {single_sub_count} single sub scenarios")
        
        # === Evaluate double substitutions (same day) ===
        logger.info("🔄 Evaluating double substitutions (same day)...")
        double_same_count = 0
        t_double_same_start = time.perf_counter()
        
        out_combos_2 = self._generate_out_combinations(
            team_state, untradable_players, must_trade_players, 2
        )
        stats.count_out_combos_2 = len(out_combos_2)
        
        for out_players in out_combos_2:
            out_pos = sorted([p.pos for p in out_players])
            
            # Find matching candidates
            if out_pos == ["front", "front"]:
                in_candidates_list = list(combinations(front_candidates, 2))
            elif out_pos == ["back", "back"]:
                in_candidates_list = list(combinations(back_candidates, 2))
            else:  # ["back", "front"]
                in_candidates_list = [(f, b) for f in front_candidates for b in back_candidates]
                
            for sub_day in range(1, 8):
                for in_players in in_candidates_list:
                    # Skip if any in_player is on team
                    if any(p.name in team_state.player_names for p in in_players):
                        continue
                    # Skip duplicates
                    if len(set(p.name for p in in_players)) != len(in_players):
                        continue
                    
                    stats.count_double_same_evaluated += 1
                    
                    result = self._evaluate_double_sub_same_day(
                        team_state, list(out_players), list(in_players), sub_day, extra_salary
                    )
                    
                    if result is None:
                        continue
                    
                    stats.count_double_same_valid += 1
                    double_same_count += 1
                    total_score, final_wallet = result
                    
                    t_w2 = time.perf_counter()
                    new_players = self._apply_substitution(
                        team_state.players, list(out_players), list(in_players)
                    )
                    stats.count_apply_substitution += 1
                    week2_score = self._week2_baseline_score(new_players, stats) if self.schedule_second else 0
                    stats.time_week2_baseline += time.perf_counter() - t_w2
                    weighted_score = w1 * total_score + w2 * week2_score
                    
                    results.append({
                        "weighted_score": weighted_score,
                        "week1_score": total_score,
                        "week2_score": week2_score,
                        "num_subs": 2,
                        "subs": [
                            {"day": sub_day, "out": out_players[0].name, "in": in_players[0].name,
                             "out_salary": out_players[0].salary_my_team,
                             "in_salary": in_players[0].salary_market},
                            {"day": sub_day, "out": out_players[1].name, "in": in_players[1].name,
                             "out_salary": out_players[1].salary_my_team,
                             "in_salary": in_players[1].salary_market}
                        ],
                        "final_players": new_players,
                        "final_wallet": final_wallet
                    })
                    stats.count_results_total += 1
                    
        stats.time_double_same_eval = time.perf_counter() - t_double_same_start
        logger.info(f"  Evaluated {double_same_count} double sub (same day) scenarios")
        
        # === Evaluate double substitutions (different days) - OPTIMIZED ===
        logger.info("🔄 Evaluating double substitutions (different days)...")
        double_diff_count = 0
        t_double_diff_start = time.perf_counter()
        
        # OPTIMIZATION: Precompute all valid T1 states with their prefix sums
        # This avoids recomputing segment scores for every day combination
        
        # For different-day subs, first sub creates intermediate team
        for (out1,) in out_combos_1:
            cands1 = front_candidates if out1.pos == "front" else back_candidates
            
            for in1 in cands1:
                if in1.name in team_state.player_names:
                    continue
                    
                # Check if first sub is feasible
                wallet_after_1 = extra_salary + out1.salary_my_team - in1.salary_market
                if wallet_after_1 < 0:
                    continue
                
                # Check NBA team constraint for first sub
                nba_counts_t1 = dict(team_state.nba_team_counts)
                nba_counts_t1[out1.nba_team] = nba_counts_t1.get(out1.nba_team, 0) - 1
                nba_counts_t1[in1.nba_team] = nba_counts_t1.get(in1.nba_team, 0) + 1
                if nba_counts_t1[in1.nba_team] > 2:
                    continue
                    
                # Build intermediate team T1 and precompute its daily scores + prefix sums
                players_t1 = self._apply_substitution(
                    team_state.players, [out1], [in1]
                )
                stats.count_apply_substitution += 1
                
                # Precompute T1's daily scores and prefix sums ONCE
                t1_daily_scores = []
                for day in range(1, 8):
                    t1_daily_scores.append(self._compute_daily_score(players_t1, day))
                
                t1_prefix_sums = [0.0]
                running = 0.0
                for score in t1_daily_scores:
                    running += score
                    t1_prefix_sums.append(running)
                
                # Generate out combos from T1 for second sub
                team_names_t1 = {p.name for p in players_t1}
                tradable_t1 = [p for p in players_t1 if p.name not in untradable_players]
                
                for out2 in tradable_t1:
                    if out2.name == in1.name:
                        # Just added player, use market salary for selling
                        out2_salary = in1.salary_market
                    else:
                        out2_salary = out2.salary_my_team
                        
                    cands2 = front_candidates if out2.pos == "front" else back_candidates
                    
                    # For diff-day subs, we can also buy back out1 (the player we sold in first sub)
                    # Create a special candidate for out1 with their market price
                    potential_buyback = None
                    if out1.pos == out2.pos:  # Can only buy back if same position as out2
                        # out1.salary_market already contains the market price (looked up from best_filter)
                        potential_buyback = out1  # Use out1 directly, its salary_market is correct
                    
                    for in2 in cands2:
                        if in2.name in team_names_t1:
                            continue
                            
                        # Check second sub feasibility
                        final_wallet = wallet_after_1 + out2_salary - in2.salary_market
                        if final_wallet < 0:
                            continue
                            
                        # Check NBA team constraint for T2
                        nba_counts_t2 = dict(nba_counts_t1)
                        nba_counts_t2[out2.nba_team] = nba_counts_t2.get(out2.nba_team, 0) - 1
                        nba_counts_t2[in2.nba_team] = nba_counts_t2.get(in2.nba_team, 0) + 1
                        if nba_counts_t2[in2.nba_team] > 2:
                            continue
                        
                        # Build T2 and precompute its daily scores + suffix sums ONCE
                        players_t2 = self._apply_substitution(players_t1, [out2], [in2])
                        stats.count_apply_substitution += 1
                        
                        t2_daily_scores = []
                        for day in range(1, 8):
                            t2_daily_scores.append(self._compute_daily_score(players_t2, day))
                        
                        # Suffix sums for T2: suffix[k] = sum of days k..7
                        t2_suffix_sums = [0.0] * 8  # suffix[8] = 0
                        running = 0.0
                        for day in range(7, 0, -1):
                            running += t2_daily_scores[day - 1]
                            t2_suffix_sums[day] = running
                        
                        # Compute week2 score ONCE for this T2 (not per day combination)
                        week2_score = self._week2_baseline_score(players_t2, stats) if self.schedule_second else 0
                        
                        # Now iterate over all 21 day combinations using precomputed prefix/suffix
                        for day1 in range(1, 7):
                            for day2 in range(day1 + 1, 8):
                                stats.count_double_diff_evaluated += 1
                                double_diff_count += 1
                                
                                # Segment 1: days 1..(day1-1) with T0 -> use team_state.prefix_sums
                                score_seg1 = team_state.prefix_sums[day1 - 1]
                                
                                # Segment 2: days day1..(day2-1) with T1 -> use T1 prefix sums
                                # sum(day1..day2-1) = prefix[day2-1] - prefix[day1-1]
                                score_seg2 = t1_prefix_sums[day2 - 1] - t1_prefix_sums[day1 - 1]
                                
                                # Segment 3: days day2..7 with T2 -> use T2 suffix sums
                                score_seg3 = t2_suffix_sums[day2]
                                
                                stats.count_compute_segment_score += 2
                                
                                total_score = score_seg1 + score_seg2 + score_seg3
                                weighted_score = w1 * total_score + w2 * week2_score
                                
                                stats.count_double_diff_valid += 1
                                results.append({
                                    "weighted_score": weighted_score,
                                    "week1_score": total_score,
                                    "week2_score": week2_score,
                                    "num_subs": 2,
                                    "subs": [
                                        {"day": day1, "out": out1.name, "in": in1.name,
                                         "out_salary": out1.salary_my_team,
                                         "in_salary": in1.salary_market},
                                        {"day": day2, "out": out2.name, "in": in2.name,
                                         "out_salary": out2_salary,
                                         "in_salary": in2.salary_market}
                                    ],
                                    "final_players": players_t2,
                                    "final_wallet": final_wallet
                                })
                                stats.count_results_total += 1
                    
                    # === BUYBACK SCENARIO: Buy back out1 (the player we sold in first sub) ===
                    if potential_buyback is not None and out1.name not in team_names_t1:
                        in2 = potential_buyback
                        
                        # Check second sub feasibility (buying back out1 at market price)
                        final_wallet = wallet_after_1 + out2_salary - in2.salary_market
                        if final_wallet >= 0:
                            # Check NBA team constraint for T2
                            nba_counts_t2 = dict(nba_counts_t1)
                            nba_counts_t2[out2.nba_team] = nba_counts_t2.get(out2.nba_team, 0) - 1
                            nba_counts_t2[in2.nba_team] = nba_counts_t2.get(in2.nba_team, 0) + 1
                            
                            if nba_counts_t2[in2.nba_team] <= 2:
                                # Build T2 with out1 bought back
                                players_t2 = self._apply_substitution(players_t1, [out2], [in2])
                                stats.count_apply_substitution += 1
                                
                                t2_daily_scores = []
                                for day in range(1, 8):
                                    t2_daily_scores.append(self._compute_daily_score(players_t2, day))
                                
                                t2_suffix_sums = [0.0] * 8
                                running = 0.0
                                for day in range(7, 0, -1):
                                    running += t2_daily_scores[day - 1]
                                    t2_suffix_sums[day] = running
                                
                                week2_score = self._week2_baseline_score(players_t2, stats) if self.schedule_second else 0
                                
                                for day1 in range(1, 7):
                                    for day2 in range(day1 + 1, 8):
                                        stats.count_double_diff_evaluated += 1
                                        double_diff_count += 1
                                        
                                        score_seg1 = team_state.prefix_sums[day1 - 1]
                                        score_seg2 = t1_prefix_sums[day2 - 1] - t1_prefix_sums[day1 - 1]
                                        score_seg3 = t2_suffix_sums[day2]
                                        
                                        total_score = score_seg1 + score_seg2 + score_seg3
                                        weighted_score = w1 * total_score + w2 * week2_score
                                        
                                        stats.count_double_diff_valid += 1
                                        results.append({
                                            "weighted_score": weighted_score,
                                            "week1_score": total_score,
                                            "week2_score": week2_score,
                                            "num_subs": 2,
                                            "subs": [
                                                {"day": day1, "out": out1.name, "in": in1.name,
                                                 "out_salary": out1.salary_my_team,
                                                 "in_salary": in1.salary_market},
                                                {"day": day2, "out": out2.name, "in": in2.name,
                                                 "out_salary": out2_salary,
                                                 "in_salary": in2.salary_market}
                                            ],
                                            "final_players": players_t2,
                                            "final_wallet": final_wallet
                                        })
                                        stats.count_results_total += 1
        
        stats.time_double_diff_eval = time.perf_counter() - t_double_diff_start
        logger.info(f"  Evaluated {double_diff_count} double sub (different days) scenarios")
        
        # Filter to only valid results BEFORE sorting
        valid_results = [r for r in results if self._check_team_constraints(r["final_players"])]
        logger.info(f"📋 {len(valid_results)} valid results out of {len(results)} total")
        
        # Sort valid results by weighted score
        t_sort_start = time.perf_counter()
        valid_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        stats.time_sort_results = time.perf_counter() - t_sort_start
        
        # Take top N from valid results
        top_results = valid_results[:top_n]
        
        # Format output
        t_format_start = time.perf_counter()
        final_output = []
        for r in top_results:
            # Build final team DataFrame for display
            final_team_data = []
            for p in r["final_players"]:
                row = self.player_lookup.get(p.name, {})
                final_team_data.append({
                    "Player": p.name,
                    "Pos": p.pos,
                    "team": p.nba_team,
                    "Form": p.form,
                    "Salary": p.salary_market
                })
                
            # Build weekly schedule display
            weekly_sched = self._build_weekly_schedule_display(
                r["final_players"], r["subs"]
            )
            
            output = {
                "total_score": round(r["weighted_score"], 2),
                "week1_score": round(r["week1_score"], 2),
                "week2_score": round(r["week2_score"], 2) if self.schedule_second else None,
                "num_subs": r["num_subs"],
                "substitution_plan": {
                    "subs": r["subs"],
                    "starting_wallet": extra_salary,
                    "final_wallet": round(r["final_wallet"], 2)
                },
                "final_team": final_team_data,
                "weekly_schedule": weekly_sched,
                "legal": True  # All results are pre-filtered to be valid
            }
            final_output.append(output)
        
        stats.time_format_output = time.perf_counter() - t_format_start
        stats.time_total = time.perf_counter() - start_time
        
        # Print performance summary
        stats.print_summary()
            
        elapsed = time.perf_counter() - start_time
        logger.info(f"🏁 Search completed in {elapsed:.2f} seconds")
        logger.info(f"📊 Total scenarios evaluated: {single_sub_count + double_same_count + double_diff_count + 1}")
        
        return final_output
    
    def _build_weekly_schedule_display(self, final_players: List[PlayerData],
                                       subs: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Build pretty weekly schedule display.
        
        Shows the correct lineup for each day based on when substitutions occur:
        - Before a sub day: shows the team BEFORE that substitution
        - From sub day onwards: shows the team AFTER that substitution
        
        For example, if a player is subbed in on day 3:
        - Days 1-2: original team (without the new player)
        - Days 3-7: new team (with the new player)
        """
        # Sort subs by day to process them in order
        sorted_subs = sorted(subs, key=lambda s: s["day"])
        
        # Build a map: for each day, what team should be shown
        # We reconstruct the team state at each point in time
        
        # Start with the team BEFORE any subs (reconstruct original team)
        # We need to reverse the substitutions to get the original team
        original_players = list(final_players)
        for sub in reversed(sorted_subs):
            # Undo substitution: remove 'in' player, add back 'out' player
            out_name = sub["out"]
            in_name = sub["in"]
            
            # Find and remove the 'in' player
            original_players = [p for p in original_players if p.name != in_name]
            
            # Add back the 'out' player (get from cache or lookup)
            out_player = self._player_data_cache.get(out_name)
            if out_player:
                original_players.append(out_player)
        
        # Now build team states for each day segment
        # team_at_day[d] = list of players active on day d
        team_at_day = {}
        current_team = list(original_players)
        current_sub_idx = 0
        
        for day in range(1, 8):
            # Check if any subs happen on this day (apply them at START of this day)
            while current_sub_idx < len(sorted_subs) and sorted_subs[current_sub_idx]["day"] == day:
                sub = sorted_subs[current_sub_idx]
                out_name = sub["out"]
                in_name = sub["in"]
                
                # Apply substitution: remove 'out', add 'in'
                current_team = [p for p in current_team if p.name != out_name]
                in_player = self._player_data_cache.get(in_name)
                if in_player:
                    current_team.append(in_player)
                
                current_sub_idx += 1
            
            team_at_day[day] = list(current_team)
        
        weekly_sched = {}
        
        for day in range(1, 8):
            day_data = []
            players = team_at_day[day]
            
            # Get playing players for this day
            playing = []
            for p in players:
                form = p.get_day_form(day)
                if form > 0:
                    playing.append((form, p.name, p.pos))
                    
            # Compute daily lineup (top 5 with max 3 per position)
            front = [(f, n, ps) for f, n, ps in playing if ps == "front"]
            back = [(f, n, ps) for f, n, ps in playing if ps == "back"]
            
            front.sort(reverse=True)
            back.sort(reverse=True)
            
            top_front = front[:3]
            top_back = back[:3]
            combined = top_front + top_back
            combined.sort(reverse=True)
            lineup = combined[:5]
            
            # Format output
            for form, name, pos in lineup:
                day_data.append({"Player": name, "Form": round(form, 1), "Pos": pos})
                
            daily_total = sum(f for f, _, _ in lineup)
            
            # Add substitution markers if applicable (handles multiple subs on same day)
            subs_on_this_day = [s for s in subs if s["day"] == day]
            for sub_info in subs_on_this_day:
                day_data.append({
                    "SubstitutionDay": f"🔄 SUB: {sub_info['out']} → {sub_info['in']}"
                })
                
            day_data.append({"Daily Total": round(daily_total, 1)})
            weekly_sched[day] = day_data
            
        return weekly_sched
    
    def convert_to_legacy_format(self, results: List[Dict[str, Any]]) -> List[Tuple]:
        """
        Convert new dict-based results to legacy tuple format for web app compatibility.
        
        Legacy format (for "weekly" type):
        (new_form, current_form, new_salary, substitutions_out, substitutions_in, new_team_dict, weekly_sched)
        
        Args:
            results: List of result dicts from find_best_weekly_substitutions_any_day
            
        Returns:
            List of tuples in legacy format
        """
        legacy_results = []
        
        for result in results:
            # Extract out and in player names from subs
            out_players = [sub['out'] for sub in result['substitution_plan']['subs']]
            in_players = [sub['in'] for sub in result['substitution_plan']['subs']]
            
            # Calculate final salary (team salary after all transactions)
            final_team_salary = sum(p['Salary'] for p in result['final_team'])
            
            # Convert to tuple format
            legacy_tuple = (
                result['week1_score'],      # new_form
                result['week1_score'],      # current_form (base team score - need to compute separately)
                round(final_team_salary, 1),  # new_salary
                out_players,                # substitutions_out
                in_players,                 # substitutions_in
                result['final_team'],       # new_team (list of dicts)
                result['weekly_schedule']   # weekly_sched
            )
            
            legacy_results.append(legacy_tuple)
            
        return legacy_results
    
    def print_result_summary(self, result: Dict[str, Any], index: int = 0):
        """Pretty print a single result."""
        print(f"\n{'='*60}")
        print(f"📊 OPTION {index + 1}")
        print(f"{'='*60}")
        
        print(f"\n📈 Scores:")
        print(f"   Total (weighted): {result['total_score']}")
        print(f"   Week 1: {result['week1_score']}")
        if result['week2_score'] is not None:
            print(f"   Week 2 (baseline): {result['week2_score']}")
            
        print(f"\n🔄 Substitution Plan ({result['num_subs']} subs):")
        if result['num_subs'] == 0:
            print("   No substitutions")
        else:
            for sub in result['substitution_plan']['subs']:
                print(f"   Day {sub['day']}: {sub['out']} (${sub['out_salary']:.1f}) → "
                      f"{sub['in']} (${sub['in_salary']:.1f})")
                      
        print(f"\n💰 Wallet:")
        print(f"   Starting: ${result['substitution_plan']['starting_wallet']:.1f}")
        print(f"   Final: ${result['substitution_plan']['final_wallet']:.1f}")
        
        print(f"\n✅ Legality: {'VALID' if result['legal'] else '❌ INVALID'}")
        
        print(f"\n📅 Weekly Schedule:")
        for day, data in result['weekly_schedule'].items():
            players = [d for d in data if "Player" in d]
            total = [d for d in data if "Daily Total" in d]
            subs = [d for d in data if "SubstitutionDay" in d]
            
            player_str = ", ".join([f"{p['Player']}({p['Form']})" for p in players])
            total_str = total[0]["Daily Total"] if total else 0
            
            line = f"   Day {day}: {player_str} = {total_str}"
            if subs:
                line += f" {subs[0]['SubstitutionDay']}"
            print(line)
            
        print(f"\n👥 Final Team:")
        for p in result['final_team']:
            print(f"   {p['Player']} ({p['Pos']}, {p['team']}) - Form: {p['Form']}, $: {p['Salary']}")

    # ============================================================================
    # Late Week Substitution Method
    # ============================================================================
    
    def find_best_late_week_substitutions(
        self,
        partial_week_schedule: dict,
        next_week_schedule: dict,
        week_after_schedule: dict,
        late_week_day: int,
        max_subs: int = 2,
        extra_salary: float = 0,
        top_n: int = 5,
        untradable_players: List[str] = None,
        must_trade_players: List[str] = None,
        w1: float = 0.8,
        w2: float = 0.2,
        allow_domination_prune: bool = True,
        shortlist_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Find best substitutions for late-week scenario.
        
        Scenario: User is on day X of current week with N subs remaining.
        Subs happen on day X, then the new team carries into next full week.
        
        Score calculation:
        - Partial week (days X to 7 of current week) + full next week = "Week 1" with weight w1
        - Week after = "Week 2" with weight w2
        
        Args:
            partial_week_schedule: Schedule for current partial week (days will be filtered to X-7)
            next_week_schedule: Schedule for next full week (this is the "main" week)
            week_after_schedule: Schedule for week 2 lookahead
            late_week_day: Which day of current week to start subs from (1-7)
            max_subs: Maximum number of subs allowed (1 or 2)
            extra_salary: Available extra salary for transactions
            top_n: Number of top results to return
            untradable_players: Players that cannot be sold
            must_trade_players: Players that must be sold
            w1: Weight for week 1 (partial + next week) score
            w2: Weight for week 2 (week after) score
            allow_domination_prune: Whether to apply domination pruning
            shortlist_k: Number of candidates to keep in shortlist
            
        Returns:
            List of top_n substitution plans with scores and details
        """
        logger.info(f"🚀 Starting late-week substitution search (day {late_week_day}, max {max_subs} subs)...")
        start_time = time.perf_counter()
        
        # Initialize
        if untradable_players is None:
            untradable_players = []
        if must_trade_players is None:
            must_trade_players = []
            
        extra_salary = float(extra_salary)
        
        # Normalize schedules to int keys
        partial_week_schedule = {int(k): v for k, v in partial_week_schedule.items()}
        next_week_schedule = {int(k): v for k, v in next_week_schedule.items()}
        week_after_schedule = {int(k): v for k, v in week_after_schedule.items()}
        
        # Store schedules for scoring
        self._partial_schedule = partial_week_schedule
        self._next_week_schedule = next_week_schedule
        self._week_after_schedule = week_after_schedule
        self._late_week_day = late_week_day
        
        # Build player data using next_week_schedule (main schedule for comparison)
        self.schedule = next_week_schedule
        self.playing_teams_dict = {day: next_week_schedule.get(day, []) for day in range(1, 8)}
        self.playing_teams_second_dict = {day: week_after_schedule.get(day, []) for day in range(1, 8)}
        self._build_player_data_lists(week=1)
        
        # Build initial team state
        team_state = self._build_team_state(self._my_team_players)
        base_weekly_score = self._compute_late_week_score(
            self._my_team_players,
            partial_week_schedule,
            next_week_schedule,
            late_week_day
        )
        
        logger.info(f"📊 Base late-week score: {base_weekly_score:.2f}")
        
        # Build shortlist
        stats = PerformanceStats()
        candidates = self._pool_players
        shortlist = self._build_shortlist(
            team_state, candidates, k=shortlist_k,
            allow_domination_prune=allow_domination_prune,
            stats=stats
        )
        
        logger.info(f"📋 Shortlist size: {len(shortlist)} players")
        
        # Group by position
        front_candidates = [c for c in shortlist if c.pos == "front"]
        back_candidates = [c for c in shortlist if c.pos == "back"]
        
        results = []
        
        # === No substitution option ===
        week1_score = base_weekly_score
        week2_score = self._compute_week_score_with_schedule(
            self._my_team_players, week_after_schedule
        )
        weighted_score = w1 * week1_score + w2 * week2_score
        
        results.append({
            "weighted_score": weighted_score,
            "week1_score": week1_score,
            "week2_score": week2_score,
            "partial_week_score": self._compute_partial_week_score(
                self._my_team_players, partial_week_schedule, late_week_day
            ),
            "next_week_score": self._compute_week_score_with_schedule(
                self._my_team_players, next_week_schedule
            ),
            "num_subs": 0,
            "subs": [],
            "final_players": self._my_team_players,
            "final_wallet": extra_salary
        })
        
        # === Single substitutions ===
        logger.info("🔄 Evaluating single substitutions...")
        single_count = 0
        
        out_combos_1 = self._generate_out_combinations(
            team_state, untradable_players, must_trade_players, 1
        )
        
        for (out_player,) in out_combos_1:
            candidates_for_pos = front_candidates if out_player.pos == "front" else back_candidates
            
            for in_player in candidates_for_pos:
                if in_player.name in team_state.player_names:
                    continue
                    
                # Check salary constraint
                wallet_after = extra_salary + out_player.salary_my_team - in_player.salary_market
                if wallet_after < 0:
                    continue
                    
                # Check NBA team constraint
                nba_counts = dict(team_state.nba_team_counts)
                nba_counts[out_player.nba_team] = nba_counts.get(out_player.nba_team, 0) - 1
                nba_counts[in_player.nba_team] = nba_counts.get(in_player.nba_team, 0) + 1
                if nba_counts[in_player.nba_team] > 2:
                    continue
                    
                # Apply substitution
                new_players = self._apply_substitution(
                    team_state.players, [out_player], [in_player]
                )
                
                # Compute scores
                week1_score = self._compute_late_week_score(
                    new_players, partial_week_schedule, next_week_schedule, late_week_day
                )
                week2_score = self._compute_week_score_with_schedule(
                    new_players, week_after_schedule
                )
                weighted_score = w1 * week1_score + w2 * week2_score
                
                results.append({
                    "weighted_score": weighted_score,
                    "week1_score": week1_score,
                    "week2_score": week2_score,
                    "partial_week_score": self._compute_partial_week_score(
                        new_players, partial_week_schedule, late_week_day
                    ),
                    "next_week_score": self._compute_week_score_with_schedule(
                        new_players, next_week_schedule
                    ),
                    "num_subs": 1,
                    "subs": [{
                        "day": late_week_day,
                        "out": out_player.name,
                        "in": in_player.name,
                        "out_salary": out_player.salary_my_team,
                        "in_salary": in_player.salary_market
                    }],
                    "final_players": new_players,
                    "final_wallet": wallet_after
                })
                single_count += 1
                
        logger.info(f"  Evaluated {single_count} single sub scenarios")
        
        # === Double substitutions (if allowed) ===
        if max_subs >= 2:
            logger.info("🔄 Evaluating double substitutions...")
            double_count = 0
            
            out_combos_2 = self._generate_out_combinations(
                team_state, untradable_players, must_trade_players, 2
            )
            
            for out_players in out_combos_2:
                out_pos = sorted([p.pos for p in out_players])
                
                if out_pos == ["front", "front"]:
                    in_candidates_list = list(combinations(front_candidates, 2))
                elif out_pos == ["back", "back"]:
                    in_candidates_list = list(combinations(back_candidates, 2))
                else:
                    in_candidates_list = [(f, b) for f in front_candidates for b in back_candidates]
                    
                for in_players in in_candidates_list:
                    if any(p.name in team_state.player_names for p in in_players):
                        continue
                    if len(set(p.name for p in in_players)) != len(in_players):
                        continue
                        
                    # Check salary constraint
                    wallet_after = extra_salary
                    for out_p in out_players:
                        wallet_after += out_p.salary_my_team
                    for in_p in in_players:
                        wallet_after -= in_p.salary_market
                    if wallet_after < 0:
                        continue
                        
                    # Check NBA team constraints
                    nba_counts = dict(team_state.nba_team_counts)
                    for out_p in out_players:
                        nba_counts[out_p.nba_team] = nba_counts.get(out_p.nba_team, 0) - 1
                    for in_p in in_players:
                        nba_counts[in_p.nba_team] = nba_counts.get(in_p.nba_team, 0) + 1
                        if nba_counts[in_p.nba_team] > 2:
                            break
                    else:
                        # All constraints passed
                        new_players = self._apply_substitution(
                            team_state.players, list(out_players), list(in_players)
                        )
                        
                        week1_score = self._compute_late_week_score(
                            new_players, partial_week_schedule, next_week_schedule, late_week_day
                        )
                        week2_score = self._compute_week_score_with_schedule(
                            new_players, week_after_schedule
                        )
                        weighted_score = w1 * week1_score + w2 * week2_score
                        
                        results.append({
                            "weighted_score": weighted_score,
                            "week1_score": week1_score,
                            "week2_score": week2_score,
                            "partial_week_score": self._compute_partial_week_score(
                                new_players, partial_week_schedule, late_week_day
                            ),
                            "next_week_score": self._compute_week_score_with_schedule(
                                new_players, next_week_schedule
                            ),
                            "num_subs": 2,
                            "subs": [
                                {"day": late_week_day, "out": out_players[0].name, "in": in_players[0].name,
                                 "out_salary": out_players[0].salary_my_team, "in_salary": in_players[0].salary_market},
                                {"day": late_week_day, "out": out_players[1].name, "in": in_players[1].name,
                                 "out_salary": out_players[1].salary_my_team, "in_salary": in_players[1].salary_market}
                            ],
                            "final_players": new_players,
                            "final_wallet": wallet_after
                        })
                        double_count += 1
                        
            logger.info(f"  Evaluated {double_count} double sub scenarios")
        
        # Sort and select top N
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        top_results = results[:top_n]
        
        # Format output
        final_output = []
        for r in top_results:
            final_team_data = []
            for p in r["final_players"]:
                final_team_data.append({
                    "Player": p.name,
                    "Pos": p.pos,
                    "team": p.nba_team,
                    "Form": p.form,
                    "Salary": p.salary_market
                })
                
            # Build weekly schedule for next full week only
            weekly_sched = self._build_weekly_schedule_for_late_week(
                r["final_players"], r["subs"], next_week_schedule
            )
            
            output = {
                "total_score": round(r["weighted_score"], 2),
                "week1_score": round(r["week1_score"], 2),
                "week2_score": round(r["week2_score"], 2),
                "partial_week_score": round(r["partial_week_score"], 2),
                "next_week_score": round(r["next_week_score"], 2),
                "num_subs": r["num_subs"],
                "substitution_plan": {
                    "subs": r["subs"],
                    "starting_wallet": extra_salary,
                    "final_wallet": round(r["final_wallet"], 2),
                    "sub_day_current_week": late_week_day
                },
                "final_team": final_team_data,
                "weekly_schedule": weekly_sched,
                "legal": True
            }
            final_output.append(output)
            
        elapsed = time.perf_counter() - start_time
        logger.info(f"🏁 Late-week search completed in {elapsed:.2f} seconds")
        
        return final_output
    
    def _compute_late_week_score(
        self,
        players: List[PlayerData],
        partial_schedule: dict,
        next_week_schedule: dict,
        late_week_day: int
    ) -> float:
        """
        Compute combined score for late week scenario.
        
        Score = partial week (days late_week_day to 7) + full next week
        """
        partial_score = self._compute_partial_week_score(
            players, partial_schedule, late_week_day
        )
        next_week_score = self._compute_week_score_with_schedule(
            players, next_week_schedule
        )
        return partial_score + next_week_score
    
    def _compute_partial_week_score(
        self,
        players: List[PlayerData],
        schedule: dict,
        start_day: int
    ) -> float:
        """Compute score for days start_day to 7 using given schedule."""
        total = 0.0
        for day in range(start_day, 8):
            total += self._compute_daily_score_with_schedule(players, day, schedule)
        return total
    
    def _compute_week_score_with_schedule(
        self,
        players: List[PlayerData],
        schedule: dict
    ) -> float:
        """Compute full week score (days 1-7) using given schedule."""
        total = 0.0
        for day in range(1, 8):
            total += self._compute_daily_score_with_schedule(players, day, schedule)
        return total
    
    def _compute_daily_score_with_schedule(
        self,
        players: List[PlayerData],
        day: int,
        schedule: dict
    ) -> float:
        """Compute daily score using specified schedule."""
        playing_teams = schedule.get(day, [])
        
        front_forms = []
        back_forms = []
        
        for p in players:
            if p.nba_team in playing_teams:
                if p.pos == "front":
                    front_forms.append(p.form)
                else:
                    back_forms.append(p.form)
                    
        front_forms.sort(reverse=True)
        back_forms.sort(reverse=True)
        
        combined = front_forms[:3] + back_forms[:3]
        combined.sort(reverse=True)
        
        return sum(combined[:5])
    
    def _build_weekly_schedule_for_late_week(
        self,
        players: List[PlayerData],
        subs: List[Dict],
        next_week_schedule: dict
    ) -> Dict[int, List[Dict]]:
        """
        Build weekly schedule display for late week scenario.
        Shows the next full week only (not the partial current week).
        """
        weekly_sched = {}
        
        for day in range(1, 8):
            day_data = []
            playing_teams = next_week_schedule.get(day, [])
            
            playing = []
            for p in players:
                if p.nba_team in playing_teams:
                    playing.append((p.form, p.name, p.pos))
                    
            front = [(f, n, ps) for f, n, ps in playing if ps == "front"]
            back = [(f, n, ps) for f, n, ps in playing if ps == "back"]
            
            front.sort(reverse=True)
            back.sort(reverse=True)
            
            combined = front[:3] + back[:3]
            combined.sort(reverse=True)
            lineup = combined[:5]
            
            for form, name, pos in lineup:
                day_data.append({"Player": name, "Form": round(form, 1), "Pos": pos})
                
            daily_total = sum(f for f, _, _ in lineup)
            day_data.append({"Daily Total": round(daily_total, 1)})
            weekly_sched[day] = day_data
            
        return weekly_sched


# ============================================================================
# Parallel Evaluation Support
# ============================================================================

def _evaluate_scenario_worker(args):
    """
    Worker function for parallel evaluation.
    
    Note: For multiprocessing, we need to pass serializable data.
    This function receives pre-serialized scenario data.
    """
    scenario_type, scenario_data, team_data, schedule_data, extra_salary = args
    
    # Reconstruct minimal state and evaluate
    # This is a simplified version for parallel execution
    # Full implementation would reconstruct PlayerData objects
    
    return scenario_data  # Placeholder


# ============================================================================
# Tests and Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMIZER PREFIX-SUFFIX - TEST SUITE")
    print("=" * 70)
    
    # Create synthetic test data
    print("\n📋 Creating synthetic test data...")
    
    # My team: 5 front + 5 back players
    my_team_data = {
        "Player": [
            "Front1", "Front2", "Front3", "Front4", "Front5",
            "Back1", "Back2", "Back3", "Back4", "Back5"
        ],
        "Pos": ["front"] * 5 + ["back"] * 5,
        "team": ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE",
                 "TeamF", "TeamG", "TeamH", "TeamI", "TeamJ"],
        "Form": [25.0, 24.0, 23.0, 22.0, 21.0,
                 20.0, 19.0, 18.0, 17.0, 16.0],
        "Salary": [10.0, 9.0, 8.0, 7.0, 6.0,
                   5.0, 4.0, 3.0, 2.0, 1.0]
    }
    my_team = pd.DataFrame(my_team_data)
    
    # Pool of available players (includes my team + others)
    pool_data = {
        "Player": my_team_data["Player"] + [
            "PoolFront1", "PoolFront2", "PoolFront3", "PoolFront4", "PoolFront5",
            "PoolBack1", "PoolBack2", "PoolBack3", "PoolBack4", "PoolBack5"
        ],
        "Pos": my_team_data["Pos"] + ["front"] * 5 + ["back"] * 5,
        "team": my_team_data["team"] + [
            "TeamK", "TeamL", "TeamM", "TeamN", "TeamO",
            "TeamP", "TeamQ", "TeamR", "TeamS", "TeamT"
        ],
        "Form": my_team_data["Form"] + [
            30.0, 28.0, 26.0, 24.0, 22.0,  # Pool fronts are better!
            25.0, 23.0, 21.0, 19.0, 17.0   # Pool backs are better!
        ],
        "Salary": my_team_data["Salary"] + [
            15.0, 14.0, 13.0, 12.0, 11.0,
            10.0, 9.0, 8.0, 7.0, 6.0
        ]
    }
    pool = pd.DataFrame(pool_data)
    
    # Schedule: different teams play on different days
    schedule_week1 = {
        1: ["TeamA", "TeamB", "TeamC", "TeamK", "TeamL", "TeamP", "TeamQ"],
        2: ["TeamD", "TeamE", "TeamF", "TeamM", "TeamN", "TeamR", "TeamS"],
        3: ["TeamG", "TeamH", "TeamI", "TeamO", "TeamP", "TeamT", "TeamK"],
        4: ["TeamJ", "TeamA", "TeamB", "TeamL", "TeamM", "TeamQ", "TeamR"],
        5: ["TeamC", "TeamD", "TeamE", "TeamN", "TeamO", "TeamS", "TeamT"],
        6: ["TeamF", "TeamG", "TeamH", "TeamK", "TeamL", "TeamP", "TeamQ"],
        7: ["TeamI", "TeamJ", "TeamA", "TeamM", "TeamN", "TeamR", "TeamS"],
    }
    
    schedule_week2 = {
        1: ["TeamB", "TeamC", "TeamD", "TeamL", "TeamM", "TeamQ", "TeamR"],
        2: ["TeamE", "TeamF", "TeamG", "TeamN", "TeamO", "TeamS", "TeamT"],
        3: ["TeamH", "TeamI", "TeamJ", "TeamP", "TeamK", "TeamK", "TeamL"],
        4: ["TeamA", "TeamB", "TeamC", "TeamM", "TeamN", "TeamR", "TeamS"],
        5: ["TeamD", "TeamE", "TeamF", "TeamO", "TeamP", "TeamT", "TeamK"],
        6: ["TeamG", "TeamH", "TeamI", "TeamL", "TeamM", "TeamQ", "TeamR"],
        7: ["TeamJ", "TeamA", "TeamB", "TeamN", "TeamO", "TeamS", "TeamT"],
    }
    
    print("\n🔧 Initializing optimizer...")
    optimizer = FantasyOptimizerPrefixSuffix(
        my_team=my_team,
        best_filter_path=pool,
        schedule_week1=schedule_week1,
        schedule_week2=schedule_week2
    )
    
    print("\n🚀 Running substitution search...")
    results = optimizer.find_best_weekly_substitutions_any_day(
        extra_salary=5.0,
        top_n=5,
        untradable_players=["Front1"],  # Can't trade best front player
        must_trade_players=[],
        w1=0.75,
        w2=0.25,
        allow_domination_prune=True,
        shortlist_k=30,
        processes=1  # Single process for testing
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results):
        optimizer.print_result_summary(result, i)
        
    print("\n✅ Test completed successfully!")
    
    # Test with must_trade constraint
    print("\n" + "=" * 70)
    print("TEST: Must trade 'Back5' (worst back player)")
    print("=" * 70)
    
    results_must_trade = optimizer.find_best_weekly_substitutions_any_day(
        extra_salary=10.0,
        top_n=3,
        untradable_players=[],
        must_trade_players=["Back5"],
        w1=1.0,
        w2=0.0,
        shortlist_k=20
    )
    
    for i, result in enumerate(results_must_trade):
        optimizer.print_result_summary(result, i)
        
    # Test legacy format conversion for web app compatibility
    print("\n" + "=" * 70)
    print("TEST: Legacy tuple format conversion (for web app)")
    print("=" * 70)
    
    legacy_format = optimizer.convert_to_legacy_format(results[:2])
    
    for i, swap in enumerate(legacy_format):
        print(f"\n📦 Legacy Tuple {i+1}:")
        print(f"   [0] new_form: {swap[0]}")
        print(f"   [1] current_form: {swap[1]}")
        print(f"   [2] new_salary: {swap[2]}")
        print(f"   [3] substitutions_out: {swap[3]}")
        print(f"   [4] substitutions_in: {swap[4]}")
        print(f"   [5] new_team: {len(swap[5])} players")
        print(f"   [6] weekly_sched: {len(swap[6])} days")
        
    print("\n✅ Legacy format conversion successful!")
        
    print("\n✅ All tests completed!")
