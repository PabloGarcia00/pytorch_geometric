import duckdb
import pandas as pd
import numpy as np
import datetime

"""
DUCKDB FOR EARNe: SYNTAX TUTORIAL
DuckDB is an in-process SQL OLAP database. In this project, we use it 
to process millions of rows of Parquet data into "wide" matrices 
in seconds.
"""

# Create an in-memory database for this tutorial
con = duckdb.connect(database=':memory:')

# ==========================================
# 1. READ_PARQUET & HIVE PARTITIONING
# ==========================================
# DuckDB can read folders of Parquet files as if they were a single table.
# 'hive_partitioning=true' means it looks at folder names (e.g., /year=2023/) 
# and automatically adds them as columns to your data.

# SYNTAX: 
# SELECT * FROM read_parquet('path/**/*.parquet', hive_partitioning=true)

# ------------------------------------------
# MOCK DATA FOR DEMO
# ------------------------------------------
df = pd.DataFrame({
    'timestamp': pd.to_datetime(['2023-01-01 12:00', '2023-01-01 12:05', '2023-01-01 12:00', '2023-01-01 12:05']),
    'MAC': ['Node_A', 'Node_A', 'Node_B', 'Node_B'],
    'net_demand': [100.5, 110.0, 50.0, 55.0]
})
con.register('raw_energy', df)

# ==========================================
# 2. TIME_BUCKET (Resampling)
# ==========================================
# This is a DuckDB-specific function. It "snaps" timestamps to a grid.
# It is much faster than Pandas' resample().

print("\n--- 2. Resampling with TIME_BUCKET ---")
query = """
    SELECT 
        time_bucket(INTERVAL '15 minutes', timestamp) as bucket,
        MAC,
        avg(net_demand) as avg_net
    FROM raw_energy
    GROUP BY 1, 2
"""
print(con.execute(query).df())
# Think: 'GROUP BY 1, 2' refers to the 1st (bucket) and 2nd (MAC) columns.


# ==========================================
# 3. PIVOT (The "Wide Matrix" Secret)
# ==========================================
# In EARNe, we need nodes as COLUMNS for the GNN. 
# DuckDB's PIVOT is extremely powerful.

print("\n--- 3. PIVOTing into Wide Matrices ---")
# This turns Nodes (Rows) into Columns.
query = """
    PIVOT raw_energy 
    ON MAC 
    USING first(net_demand) 
    GROUP BY timestamp 
    ORDER BY timestamp
"""
wide_df = con.execute(query).df()
print(wide_df)
# Think: This shape [Timestamps, Nodes] is exactly what our Master Bundle needs.


# ==========================================
# 4. COALESCE & JOINS (Imputation)
# ==========================================
# COALESCE returns the first non-NULL value in its arguments.
# We use this for our 3-tier weather fallback.

print("\n--- 4. COALESCE Fallback Logic ---")
# Mock: Local Station has NULL, but Fleet Avg has 22.5
query = """
    SELECT 
        COALESCE(NULL, 22.5, 0) as temp_tier_2,
        COALESCE(18.0, 22.5, 0) as temp_tier_1,
        COALESCE(NULL, NULL, 0) as temp_tier_3
"""
print(con.execute(query).df())
# Logic: 
# Tier 1: Local Station (if exists)
# Tier 2: Fleet Average (if station fails)
# Tier 3: 0 (if all fails)


# ==========================================
# 5. DUCKDB -> PANDAS -> PYTORCH
# ==========================================
# Why use DuckDB first?
# 1. Memory: It doesn't load the whole file; it streams it.
# 2. Speed: It uses all CPU cores for the PIVOT and AGGREGATE.
# 3. Type Safety: It handles Parquet schemas strictly.

def duck_to_torch():
    # Execute SQL -> Get Pandas -> Convert to Torch
    df = con.execute("SELECT net_demand FROM raw_energy").df()
    tensor = torch.tensor(df.values)
    return tensor

print("\nTutorial Complete. Check 'custom_graphgym/loader/graph_dataset.py' to see these combined.")
