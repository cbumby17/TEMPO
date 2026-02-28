from pathlib import Path
import pandas as pd


def load_example_data() -> pd.DataFrame:
    """
    Load the bundled example perturbation-response dataset.

    Returns a simulated longitudinal compositional dataset with 40 subjects
    (15 responders, 25 non-responders) measured over 12 timepoints across
    15 features. Features 0–2 carry a known trajectory motif in the responder
    group during timepoints 3–8, representing a coordinated feature response
    to a perturbation.

    Ground truth is stored in df.attrs for use with
    simulate.get_ground_truth() and simulate.evaluation_report().

    Returns
    -------
    pd.DataFrame
        Long-format with columns: subject_id, timepoint, feature, value, outcome.
        outcome=1 indicates responders; outcome=0 indicates non-responders.
    """
    data_path = Path(__file__).parent / "data" / "example_longitudinal.csv"
    df = pd.read_csv(data_path)
    df.attrs = {
        "motif_features": ["feature_000", "feature_001", "feature_002"],
        "motif_window": (3, 8),
        "n_cases": 15,
        "n_controls": 25,
        "motif_strength": 2.5,
    }
    return df
