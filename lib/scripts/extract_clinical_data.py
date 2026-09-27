"""Extract and standardize clinical data for each cohort."""

import gzip
import json
import os
import pandas as pd

from genolib import DATA_DIR as _DATA_DIR
DATA_DIR = str(_DATA_DIR)  # central store (absolute), works from any cwd
RAW_DIR = f"{DATA_DIR}/raw/gastric"
GEO_CLINICAL_DIR = f"{DATA_DIR}/clinical/gastric/GEO"
OUT_DIR = f"{DATA_DIR}/clinical/gastric/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SENTINEL = "'--"  # GDC missing value marker


# TCGA STAD

def extract_tcga():
    """Combine GDC clinical files + cBioPortal data → standardized TCGA table."""

    # --- GDC clinical.tsv (broad cohort, n~443) ---
    gdc_path = f"{DATA_DIR}/clinical/gastric/TCGA/clinical.tsv"
    gdc = pd.read_csv(gdc_path, sep="\t", low_memory=False)

    def clean(val):
        return None if str(val).strip() in (SENTINEL, "nan", "", "--") else val

    rows = []
    for _, r in gdc.drop_duplicates("cases.submitter_id").iterrows():
        pid = r["cases.submitter_id"]
        vital = r.get("demographic.vital_status", "")
        days_death = clean(r.get("demographic.days_to_death"))
        days_fu = clean(r.get("diagnoses.days_to_last_follow_up"))

        # OS in months: use days_to_death for dead, days_to_last_follow_up for alive
        os_days = None
        os_status = None
        if str(vital) == "Dead":
            os_status = 1
            os_days = float(days_death) if days_death is not None else (
                float(days_fu) if days_fu is not None else None)
        elif str(vital) == "Alive":
            os_status = 0
            os_days = float(days_fu) if days_fu is not None else None

        os_months = round(os_days / 30.44, 2) if os_days is not None else None

        rows.append({
            "patient_id": pid,
            "os_months": os_months,
            "os_status": os_status,
            "age": clean(r.get("demographic.age_at_index")),
            "sex": str(r.get("demographic.gender", "")).capitalize() or None,
            "stage": clean(r.get("diagnoses.ajcc_pathologic_stage")),
        })

    tcga = pd.DataFrame(rows)

    # --- cBioPortal stad_tcga_pub (molecular subtypes, Lauren, MSI, EBV, n~295) ---
    pub_path = f"{GEO_CLINICAL_DIR}/TCGA_STAD_cbioportal_patient.json"
    if os.path.exists(pub_path):
        with open(pub_path) as f:
            pub_data = json.load(f)
        pub_df = pd.DataFrame(pub_data)[["patientId", "clinicalAttributeId", "value"]]
        pub_wide = pub_df.pivot_table(
            index="patientId", columns="clinicalAttributeId",
            values="value", aggfunc="first"
        ).reset_index().rename(columns={"patientId": "patient_id"})

        mol_cols = {
            "LAUREN_CLASS": "lauren",
            "MSI_STATUS": "msi_status",
            "EBV_PRESENT": "ebv_status",
            "MOLECULAR_SUBTYPE": "molecular_subtype",
            "OS_MONTHS": "os_months_pub",
            "OS_STATUS": "os_status_pub",
            "DFS_MONTHS": "dfs_months",
            "DFS_STATUS": "dfs_status",
        }
        pub_subset = pub_wide[
            ["patient_id"] + [c for c in mol_cols if c in pub_wide.columns]
        ].rename(columns=mol_cols)
        tcga = tcga.merge(pub_subset, on="patient_id", how="left")

        # Fill OS from pub where GDC is missing
        if "os_months_pub" in tcga.columns:
            mask = tcga["os_months"].isna() & tcga["os_months_pub"].notna()
            tcga.loc[mask, "os_months"] = pd.to_numeric(
                tcga.loc[mask, "os_months_pub"], errors="coerce")
            tcga.drop(columns=["os_months_pub"], inplace=True)
        if "os_status_pub" in tcga.columns:
            mask = tcga["os_status"].isna() & tcga["os_status_pub"].notna()
            tcga.loc[mask, "os_status"] = tcga.loc[mask, "os_status_pub"].map(
                {"1:DECEASED": 1, "0:LIVING": 0})
            tcga.drop(columns=["os_status_pub"], inplace=True)

    # Map OS_STATUS text → 0/1 if not already numeric
    if tcga["os_status"].dtype == object:
        tcga["os_status"] = tcga["os_status"].map(
            {"1:DECEASED": 1, "0:LIVING": 0, 1: 1, 0: 0})

    tcga["cohort"] = "TCGA"
    out = f"{OUT_DIR}/TCGA_clinical.csv"
    tcga.to_csv(out, index=False)
    print(f"TCGA: {len(tcga)} patients → {out}")
    print(f"  OS available: {tcga['os_months'].notna().sum()}")
    return tcga


# GSE84437, extract from SOFT file

def extract_gse84437():
    path = f"{RAW_DIR}/GSE84437_family.soft.gz"
    records = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        cur = {}
        for line in f:
            if line.startswith("^SAMPLE"):
                if cur:
                    records.append(cur)
                cur = {"gsm_id": line.split("=")[1].strip()}
            elif line.startswith("!Sample_title"):
                cur["sample_title"] = line.split("=")[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[1].strip().strip('"')
                if ":" in val:
                    k, v = val.split(":", 1)
                    cur[k.strip()] = v.strip()
        if cur:
            records.append(cur)

    df = pd.DataFrame(records)
    df = df[df["tissue"].str.lower().str.contains("cancer|tumor", na=False)].copy()

    out_df = pd.DataFrame({
        "patient_id": df["sample_title"],
        "gsm_id": df["gsm_id"],
        "os_months": pd.to_numeric(df["duration overall survival"], errors="coerce"),
        "os_status": pd.to_numeric(df["death"], errors="coerce"),
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": df["Sex"].str.capitalize(),
        "pt_stage": df["ptstage"],
        "pn_stage": df["pnstage"],
        "cohort": "GSE84437",
    })

    out = f"{OUT_DIR}/GSE84437_clinical.csv"
    out_df.to_csv(out, index=False)
    print(f"GSE84437: {len(out_df)} patients → {out}")
    print(f"  OS available: {out_df['os_months'].notna().sum()}")
    return out_df


# GSE15459 + GSE34942, from GEO outcome Excel files

def extract_geo_outcome_xls(xls_path, cohort_name):
    df = pd.read_excel(xls_path)

    # Standardise column names
    col_map = {
        "GSM ID": "gsm_id",
        "ID": "sample_id",
        "Subtype": "molecular_subtype",
        "Age_at_surgery": "age",
        "Gender": "sex",
        "Laurenclassification": "lauren",
        "Stage": "stage",
        "Overall.Survival (Months)**": "os_months",
        "Outcome (1=dead)": "os_status",
    }
    df = df.rename(columns=col_map)

    # Clean up GSM IDs, strip any whitespace
    df["gsm_id"] = df["gsm_id"].astype(str).str.strip()
    df["patient_id"] = df["gsm_id"]
    df["os_status"] = pd.to_numeric(df["os_status"], errors="coerce")
    df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce").astype("Int64")
    df["cohort"] = cohort_name

    out = f"{OUT_DIR}/{cohort_name}_clinical.csv"
    df.to_csv(out, index=False)
    print(f"{cohort_name}: {len(df)} patients → {out}")
    print(f"  OS available: {df['os_months'].notna().sum()}")
    return df


# GSE62254 / GSE66229 (ACRG), from NatComm 2018 supplementary data

def extract_acrg():
    """
    Extract ACRG clinical data from the Nature Communications 2018 paper
    supplementary (Sohn et al., doi:10.1038/s41467-018-04179-8).
    File: data/clinical/gastric/GEO/ACRG_clinical_NatComm2018.xlsx, sheet 'ACRG'
    The GEO_ID column contains GSM IDs that match our expression matrices.
    """
    supp_path = f"{GEO_CLINICAL_DIR}/ACRG_clinical_NatComm2018.xlsx"
    if not os.path.exists(supp_path):
        print("ACRG: NatComm supplement not found, skipping")
        return None

    df = pd.read_excel(supp_path, sheet_name="ACRG")

    # Standardise column names
    df = df.rename(columns={
        "GEO_ID": "gsm_id",
        "SCRI No.": "patient_num",
        "Sample\nName": "sample_name",
        "ACRG.sub": "molecular_subtype",
        "Death": "os_status",
        "OS.m": "os_months",
        "DFS.m": "dfs_months",
        "Recur": "recurrence",
        "sex": "sex",
        "age": "age",
        "Lauren": "lauren",
        "Stage": "stage",
        "pStage": "p_stage",
    })

    # EBV column has a long multi-line header, find it by partial match
    ebv_col = next((c for c in df.columns if "EBV" in str(c)), None)
    if ebv_col:
        df["ebv_status"] = df[ebv_col].map({0: "Negative", 1: "Positive"})

    df["gsm_id"] = df["gsm_id"].astype(str).str.strip()
    df["patient_id"] = df["gsm_id"]
    df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
    df["os_status"] = pd.to_numeric(df["os_status"], errors="coerce")
    df["cohort"] = "GSE62254"

    keep_cols = [
        "patient_id", "gsm_id", "sample_name", "os_months", "os_status",
        "dfs_months", "recurrence", "age", "sex", "lauren", "stage", "p_stage",
        "molecular_subtype", "ebv_status", "cohort",
    ]
    out_df = df[[c for c in keep_cols if c in df.columns]].copy()
    out = f"{OUT_DIR}/GSE62254_clinical.csv"
    out_df.to_csv(out, index=False)
    print(f"ACRG (GSE62254): {len(out_df)} patients, OS={out_df['os_months'].notna().sum()} -> {out}")
    return out_df


# GSE35809, subtypes in SOFT, no survival; extract what's available

def extract_gse35809():
    path = f"{RAW_DIR}/GSE35809_family.soft.gz"
    records = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        cur = {}
        for line in f:
            if line.startswith("^SAMPLE"):
                if cur:
                    records.append(cur)
                cur = {"gsm_id": line.split("=")[1].strip()}
            elif line.startswith("!Sample_title"):
                cur["sample_title"] = line.split("=")[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[1].strip().strip('"')
                if ":" in val:
                    k, v = val.split(":", 1)
                    cur[k.strip()] = v.strip()
        if cur:
            records.append(cur)

    df = pd.DataFrame(records)
    df = df[df["tissue"].str.lower().str.contains("cancer|tumor|gastric", na=False)].copy()

    out_df = pd.DataFrame({
        "patient_id": df["gsm_id"],
        "gsm_id": df["gsm_id"],
        "sample_title": df["sample_title"],
        "molecular_subtype": df.get("subtype"),
        "cohort": "GSE35809",
    })
    out = f"{OUT_DIR}/GSE35809_clinical.csv"
    out_df.to_csv(out, index=False)
    print(f"GSE35809: {len(out_df)} samples → {out} (subtypes only, no OS)")
    return out_df


# Minimal metadata for remaining cohorts (tissue type only)

def extract_minimal(gse, cohort_name=None):
    cohort_name = cohort_name or gse
    path = f"{RAW_DIR}/{gse}_family.soft.gz"
    if not os.path.exists(path):
        print(f"{cohort_name}: SOFT file not found, skipping")
        return None

    records = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        cur = {}
        for line in f:
            if line.startswith("^SAMPLE"):
                if cur:
                    records.append(cur)
                cur = {"gsm_id": line.split("=")[1].strip()}
            elif line.startswith("!Sample_title"):
                cur["sample_title"] = line.split("=")[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[1].strip().strip('"')
                if ":" in val:
                    k, v = val.split(":", 1)
                    cur[k.strip()] = v.strip()
        if cur:
            records.append(cur)

    df = pd.DataFrame(records)
    df["patient_id"] = df["gsm_id"]
    df["cohort"] = cohort_name
    out = f"{OUT_DIR}/{cohort_name}_clinical.csv"
    df.to_csv(out, index=False)
    print(f"{cohort_name}: {len(df)} samples → {out} (minimal metadata only)")
    return df


# GSE66229, tumor samples reuse GSE62254 clinical; normal samples flagged

def build_gse66229():
    expr = pd.read_csv(
        f"{DATA_DIR}/processed/gastric/ACRG_GSE66229_for_xcell.txt",
        sep="\t", index_col=0, nrows=0,
    )
    all_ids = list(expr.columns)

    # Pull tumor clinical from GSE62254
    clin62 = pd.read_csv(f"{OUT_DIR}/GSE62254_clinical.csv")
    tumor_ids = set(clin62["gsm_id"].astype(str))

    normal_rows = [
        {"patient_id": gsm, "gsm_id": gsm, "tissue_type": "Normal", "cohort": "GSE66229"}
        for gsm in all_ids if gsm not in tumor_ids
    ]

    tumor_df = clin62[clin62["gsm_id"].isin(all_ids)].copy()
    tumor_df["tissue_type"] = "Tumor"
    tumor_df["cohort"] = "GSE66229"

    out_df = pd.concat([tumor_df, pd.DataFrame(normal_rows)], ignore_index=True)
    out = f"{OUT_DIR}/GSE66229_clinical.csv"
    out_df.to_csv(out, index=False)
    n_tumor = tumor_df.shape[0]
    n_normal = len(normal_rows)
    print(f"GSE66229: {n_tumor} tumor + {n_normal} normal -> {out}")
    return out_df


# Filter all clinical files to patients present in expression matrices

def filter_to_expression():
    import glob, re

    expr_files = sorted(glob.glob(f"{DATA_DIR}/processed/gastric/*.txt"))

    # Build map: gse -> set of sample IDs in expression matrix
    expr_ids_map = {}
    for f in expr_files:
        fname = os.path.basename(f)
        if "TCGA" in fname:
            key = "TCGA"
        else:
            m = re.search(r"(GSE\d+)", fname)
            key = m.group(1) if m else None
        if key is None:
            continue
        df = pd.read_csv(f, sep="\t", index_col=0, nrows=0)
        expr_ids_map[key] = set(df.columns)

    for gse, expr_ids in expr_ids_map.items():
        clin_path = f"{OUT_DIR}/{gse}_clinical.csv"
        if not os.path.exists(clin_path):
            continue

        clin = pd.read_csv(clin_path)
        id_col = "gsm_id" if "gsm_id" in clin.columns else "patient_id"

        before = len(clin)
        clin = clin[clin[id_col].astype(str).str.strip().isin(expr_ids)].copy()
        after = len(clin)

        clin.to_csv(clin_path, index=False)
        if before != after:
            print(f"  {gse}: {before} -> {after} rows (dropped {before - after})")
        else:
            print(f"  {gse}: {after} rows (no change)")


# Main

if __name__ == "__main__":
    print("=" * 60)
    print("Extracting clinical data for all gastric cancer cohorts")
    print("=" * 60)

    print("\n--- TCGA STAD ---")
    extract_tcga()

    print("\n--- GSE84437 ---")
    extract_gse84437()

    print("\n--- GSE15459 (Singapore A) ---")
    extract_geo_outcome_xls(f"{GEO_CLINICAL_DIR}/GSE15459_outcome.xls", "GSE15459")

    print("\n--- GSE34942 (Singapore B) ---")
    extract_geo_outcome_xls(f"{GEO_CLINICAL_DIR}/GSE34942_outcome.xls", "GSE34942")

    print("\n--- ACRG (GSE62254 / GSE66229) ---")
    extract_acrg()

    print("\n--- GSE35809 (Australian) ---")
    extract_gse35809()

    print("\n--- Minimal metadata for remaining cohorts ---")
    for gse in ["GSE51105", "GSE54129", "GSE57303", "GSE118916"]:
        extract_minimal(gse)

    print("\n--- GSE66229 (normal + tumor) ---")
    build_gse66229()

    print("\n--- Filter all clinical files to expression matrix samples ---")
    filter_to_expression()

    print("\n--- Summary ---")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".csv"):
            df = pd.read_csv(f"{OUT_DIR}/{f}")
            os_col = "os_months" if "os_months" in df.columns else None
            os_n = df["os_months"].notna().sum() if os_col else "N/A"
            print(f"  {f}: {len(df)} rows, OS n={os_n}")
