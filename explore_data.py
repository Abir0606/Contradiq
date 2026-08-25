import pandas as pd
from pathlib import Path

# Load the master clauses CSV — this has one row per contract,
# with columns for each of the 41 clause categories
df = pd.read_csv("D:\\Contradiq\\data\\raw\\CUAD_v1\\master_clauses.csv")

print(f"Number of contracts: {len(df)}")
print(f"Number of columns (clause categories + metadata): {len(df.columns)}")
print("\nFirst few column names:")
print(df.columns[:10].tolist())

print("\nSample contract filename:")
print(df["Filename"].iloc[0])

# Check the raw contract files exist
contract_dir = Path("D:\\Contradiq\\data\\raw\\CUAD_v1\\full_contract_pdf")
if not contract_dir.exists():
    contract_dir = Path("D:\\Contradiq\\data\\raw\\CUAD_v1\\full_contract_txt")
print(f"\nLooking for contracts in: {contract_dir}")
print(f"Number of contract files found: {len(list(contract_dir.rglob('*')))}")