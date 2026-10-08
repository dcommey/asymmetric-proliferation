"""Compare the window search with a dense grid for points of the reference design.

You can run this script from any directory. Install the package dependencies
before you run it. The check is numerical. It does not prove global optimality.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asymprolif.experiments import scan_calibration, scan_point
from asymprolif.model import Calibration, evaluate_policy, prerelease_outcome


def main() -> None:
    samples, intervals = 128, 4000
    maximum_shortfall = 0.0
    for sample_id in range(1, samples + 1):
        calibration = scan_calibration(Calibration(), scan_point(sample_id))
        optimized = evaluate_policy(calibration, "prerelease").welfare
        dense = max(prerelease_outcome(calibration, 2 * j / intervals).welfare
                    for j in range(intervals + 1))
        maximum_shortfall = max(maximum_shortfall, dense - optimized)
    print(json.dumps({
        "design_points_checked": samples,
        "dense_windows_per_point": intervals + 1,
        "maximum_optimizer_shortfall": maximum_shortfall,
    }, indent=2))
    if maximum_shortfall > 1e-8:
        raise SystemExit("Optimizer fell below the dense screen.")


if __name__ == "__main__":
    main()
