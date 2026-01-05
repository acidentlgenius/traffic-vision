import pandas as pd
import json
import logging
import scipy.stats as stats
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

INFERENCE_LOG_PATH = "inference.log"
REPORT_PATH = "reports/drift_report.json"
os.makedirs("reports", exist_ok=True)

def load_logs():
    """Parses inference.log into a Pandas DataFrame."""
    data = []
    if not os.path.exists(INFERENCE_LOG_PATH):
        return pd.DataFrame()
        
    with open(INFERENCE_LOG_PATH, "r") as f:
        for line in f:
            if "{" in line:
                try:
                    # Extract JSON part
                    json_str = line[line.find("{"):]
                    entry = json.loads(json_str)
                    data.append(entry)
                except:
                    continue
    
    df = pd.DataFrame(data)
    return df

def run_drift_check():
    df = load_logs()
    if df.empty:
        logging.warning("No logs found. Cannot check for drift.")
        return

    # Simulate Reference Data (High confidence, stable)
    # Synthetic reference: mean cons ~ 0.8
    reference_conf = [0.8, 0.85, 0.75, 0.9, 0.82] * 20
    
    # Current data
    if "mean_confidence" not in df.columns:
        logging.error("mean_confidence column missing in logs")
        return

    current_conf = df["mean_confidence"].dropna().tolist()
    
    # Check 1: KS-Test on Confidence Scores
    # Tests if the two samples are drawn from the same distribution
    statistic, p_value = stats.ks_2samp(reference_conf, current_conf)
    
    drift_detected = p_value < 0.05 # Standard threshold
    
    logging.info(f"KS Statistic: {statistic:.4f}, P-Value: {p_value:.10f}")
    
    result = {
        "drift_detected": bool(drift_detected),
        "metric": "KS_Test_Confidence",
        "p_value": float(p_value),
        "threshold": 0.05,
        "current_mean_conf": float(pd.Series(current_conf).mean()),
        "reference_mean_conf": float(pd.Series(reference_conf).mean())
    }
    
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)
        
    logging.info("Drift check complete.")
    logging.info(json.dumps(result, indent=2))
    
    if drift_detected:
        print("DRIFT_DETECTED")
        
    return drift_detected

if __name__ == "__main__":
    run_drift_check()
