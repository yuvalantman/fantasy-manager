# Fantasy Basketball Optimizer Algorithm

## The Problem

You have a fantasy basketball team of **10 players** and want to find the best way to make **0, 1, or 2 substitutions** during a 7-day gameweek to maximize your total score.

### Key Constraints

1. **Team Structure**: 10 players, each is either "front" (F, G, C) or "back" (other positions)
2. **Daily Lineup**: Each day, only 5 players score points. The top 5 are selected from those playing that day, with a maximum of 3 from any single position (front or back).
3. **Salary Cap**: You have a budget. When you sell a player, you get their sale price. When you buy, you pay their market price. Budget must never go negative.
4. **NBA Team Limit**: Maximum 2 players from the same NBA team at any time.
5. **Substitution Timing**: A substitution takes effect at the START of a given day. If you sub on day 3, the new player contributes from day 3 onwards.
6. **Two-Week Lookahead**: We weight current week (w1) and next week (w2) scores for strategic planning.

### What We're Optimizing

Find the substitution plan (0, 1, or 2 subs on any days) that maximizes:

```
Weighted Score = w1 × Week1_Score + w2 × Week2_Score
```

Where Week1_Score is the sum of daily lineup scores across all 7 days.

---

## The Challenge: Combinatorial Explosion

### Scenario Counts

| Scenario | Formula | Approximate Count |
|----------|---------|-------------------|
| No subs | 1 | 1 |
| 1 sub (any day) | 10 × 50 × 7 = 3,500 | ~3,500 |
| 2 subs, same day | C(10,2) × C(50,2) × 7 ≈ 45 × 1225 × 7 | ~385,000 |
| 2 subs, different days | 10 × 50 × 10 × 50 × 21 | ~5,250,000 |

With position matching and candidate filtering, actual numbers are lower, but **different-day double subs** still dominate the search space.

### The Core Expense

For each scenario, we need to compute a **weekly score**:
- Sum of 7 daily scores
- Each daily score requires sorting ~5-10 playing players by form

Naively, evaluating 5 million scenarios × 7 days × sorting = **very slow**.

---

## The Solution: Prefix-Suffix Optimization

### Key Insight

For different-day substitutions with days `(day1, day2)` where `day1 < day2`:

```
Week consists of 3 segments:
├── Segment 1: days 1 to (day1-1)  →  Original team T0
├── Segment 2: days day1 to (day2-1)  →  Team after 1st sub (T1)  
└── Segment 3: days day2 to 7  →  Team after 2nd sub (T2)
```

### The Optimization

Instead of recomputing segment scores for each of the 21 day combinations, we precompute:

1. **T0 Prefix Sums** (once per team):
   - `prefix[k]` = sum of daily scores for days 1 to k
   - Segment 1 score = `prefix[day1 - 1]`

2. **T1 Prefix Sums** (once per first substitution):
   - After each `(out1, in1)` substitution on T0
   - Compute T1's 7 daily scores once
   - Build prefix sums
   - Segment 2 score = `T1_prefix[day2 - 1] - T1_prefix[day1 - 1]`

3. **T2 Suffix Sums** (once per second substitution):
   - After each `(out2, in2)` substitution on T1
   - Compute T2's 7 daily scores once
   - Build suffix sums: `suffix[k]` = sum of days k to 7
   - Segment 3 score = `T2_suffix[day2]`

### Before vs After

**Before (naive approach):**
```
For each (out1, in1, out2, in2) combo:
    For each (day1, day2) pair (21 iterations):
        Compute 7 daily scores for the 3-segment team
        Total: 21 × 7 = 147 daily score computations
```

**After (optimized):**
```
For each (out1, in1):
    Compute T1 daily scores (7 computations)
    Build T1 prefix sums (O(7))
    
    For each (out2, in2):
        Compute T2 daily scores (7 computations)
        Build T2 suffix sums (O(7))
        Compute week2 score ONCE
        
        For each (day1, day2) pair (21 iterations):
            Segment scores = O(1) lookups
            Total: 3 array lookups
```

**Speedup**: From ~147 computations to ~14 computations per player combination = **~10x faster**

---

## Algorithm Pseudocode

```python
def find_best_substitutions():
    results = []
    
    # Precompute original team's prefix sums (T0)
    T0_prefix = compute_prefix_sums(original_team)
    
    # === Single Substitutions ===
    for out1 in team:
        for in1 in candidates:
            for sub_day in 1..7:
                if feasible(out1, in1):
                    score = T0_prefix[sub_day-1] + T1_suffix[sub_day]
                    results.append(score)
    
    # === Double Subs, Same Day ===
    for (out1, out2) in pairs(team):
        for (in1, in2) in candidate_pairs:
            for sub_day in 1..7:
                if feasible(...):
                    # Similar prefix/suffix approach
                    results.append(...)
    
    # === Double Subs, Different Days (OPTIMIZED) ===
    for out1 in team:
        for in1 in candidates:
            if not feasible(out1, in1): continue
            
            # Build T1 and its prefix sums ONCE
            T1 = apply_sub(team, out1, in1)
            T1_prefix = compute_prefix_sums(T1)
            
            for out2 in T1:
                for in2 in candidates:
                    if not feasible(out2, in2): continue
                    
                    # Build T2 and its suffix sums ONCE
                    T2 = apply_sub(T1, out2, in2)
                    T2_suffix = compute_suffix_sums(T2)
                    week2_score = compute_week2(T2)  # ONCE per T2
                    
                    # Now iterate 21 day combinations with O(1) lookups
                    for day1 in 1..6:
                        for day2 in (day1+1)..7:
                            seg1 = T0_prefix[day1 - 1]
                            seg2 = T1_prefix[day2 - 1] - T1_prefix[day1 - 1]
                            seg3 = T2_suffix[day2]
                            
                            total = seg1 + seg2 + seg3
                            weighted = w1 * total + w2 * week2_score
                            results.append(weighted)
    
    return sorted(results)[:top_n]
```

---

## Additional Optimizations

### 1. Candidate Filtering
- Only consider players with form above a threshold
- Position matching (front can only swap with front)

### 2. Domination Pruning (optional)
- Skip candidates dominated by better players at same or lower price

### 3. Shortlist Limiting
- Cap candidates per position to avoid exponential blowup

### 4. Week 2 Score Caching
- Compute `week2_score` once per final team T2, not per day combination
- This alone saves 21× computation for week2 in diff-day scenarios

---

## Complexity Analysis

Let:
- N = team size (10)
- C = candidates per position (~25-50)
- D = days in week (7)

| Phase | Complexity |
|-------|------------|
| Single subs | O(N × C × D) |
| Double same-day | O(N² × C² × D) |
| Double diff-day (naive) | O(N × C × N × C × D²) with D² = 21 |
| Double diff-day (optimized) | O(N × C × (D + N × C × D)) |

The optimization removes the D² factor from the innermost loop, replacing it with O(1) lookups.

---

## Data Structures

### PlayerData
```python
@dataclass
class PlayerData:
    name: str
    pos: str           # "front" or "back"
    nba_team: str      # e.g., "LAL"
    form: float        # season average
    salary_my_team: float   # price to sell (user's price)
    salary_market: float    # price to buy (market price)
    daily_forms: Dict[int, float]  # day -> form if playing
```

### TeamState
```python
@dataclass
class TeamState:
    players: List[PlayerData]
    player_names: Set[str]
    nba_team_counts: Dict[str, int]
    prefix_sums: List[float]  # precomputed
```

---

## Summary

The key innovation is treating the 7-day week as segments and using **prefix/suffix sums** to compute segment totals in O(1) instead of O(D). This transforms an O(D²) inner loop into O(D² × O(1)) = O(D²), where the constant factor is just 3 array lookups instead of 7 daily score computations.

Combined with computing week2 scores once per final team rather than per day combination, this yields approximately **10-20x speedup** for the most expensive phase of the algorithm.
