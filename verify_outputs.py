from pathlib import Path

OUTPUT = Path("pedestrian_behavior_simulation_outputs")
EXPECTED = [
    "synthetic_pedestrian_population.csv",
    "beta_or_check.csv",
    "scenario_results.csv",
    "monte_carlo_results.csv",
    "sensitivity_results.csv",
    "behavioral_response_surface.csv",
    "Figure_1_Scenario_Comparison.png",
    "Figure_2_Vehicle_Speed_Response.png",
    "Figure_3_Three_Wheeler_Response.png",
    "Figure_4_Behavioral_Response_Surface.png",
    "Figure_5_Monte_Carlo_Scenarios.png",
    "Figure_Sensitivity_Vehicle_Speed.png",
    "Figure_Sensitivity_Vehicle_Gap.png",
    "Figure_Sensitivity_Waiting_Time.png",
    "Figure_Sensitivity_Pedestrian_Speed.png",
    "Figure_Sensitivity_Vehicle_Encountered.png",
]

missing = [name for name in EXPECTED if not (OUTPUT / name).exists()]
if missing:
    print("Missing expected files:")
    for name in missing:
        print(f"  - {name}")
    raise SystemExit(1)

print(f"OK: found all {len(EXPECTED)} expected output files in {OUTPUT}/")
