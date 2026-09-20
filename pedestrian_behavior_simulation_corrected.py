
# ============================================================
# BEHAVIOR-INFORMED STOCHASTIC SIMULATION
# Pedestrian crossing speed-change behavior under mixed traffic
#
# IMPORTANT:
# - The original pedestrian-level SPSS dataset is NOT available.
# - Individual records generated here are SYNTHETIC.
# - Categorical probabilities come from the supplied paper.
# - Continuous-variable distributions use explicit assumptions
#   because the supplied paper gives ranges/SDs but not complete
#   distributions or means for all variables.
# - The paper's intercept is not available. An illustrative
#   intercept is calibrated to the reported 30.85% prevalence.
# - Therefore this is NOT a reconstruction of the original SPSS model.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import brentq
from pathlib import Path
import shutil

# ----------------------------
# 1. Reproducibility
# ----------------------------
np.random.seed(42)

N = 3209

# Published prevalence
TARGET_SPEED_CHANGE_RATE = 0.3085
TARGET_PATH_CHANGE_RATE = 0.2079

# Published marginal distributions
GENDER_PROBS = {"Female": 0.1963, "Male": 0.8036}
AGE_PROBS = {
    "Child": 0.0418,
    "Young": 0.4911,
    "Middle": 0.3300,
    "Elder": 0.1371,
}
GROUP_PROBS = {
    "Single": 0.4584,
    "Two": 0.2801,
    "More": 0.2615,
}
VEHICLE_PROBS = {
    "Two-wheeler": 0.2089,
    "Three-wheeler": 0.3727,
    "Four-wheeler": 0.2976,
    "Heavy": 0.1208,
}
LAG_GAP_PROBS = {"Gap": 0.2468, "Lag": 0.7532}

# Published beta coefficients from the supplied paper
BETA = {
    "Male": 0.479,
    "Young": 0.771,
    "Middle": 0.844,
    "Elder": 0.120,                 # NOTE: paper's OR is inconsistent
    "Path_Change": 2.116,
    "Two_Pedestrians": -0.540,
    "More_Pedestrians": -2.003,
    "Three_Wheeler": 0.990,
    "Four_Wheeler": 0.480,
    "Heavy_Vehicle": 0.426,
    "Vehicle_Speed": 0.037,
    "Vehicle_Gap": 0.217,
    "Pedestrian_Speed": 0.295,
    "Waiting_Time": 0.114,
    "Lag": 0.720,
    "Vehicle_Encountered": -0.033,
}

REPORTED_OR = {
    "Male": 1.614,
    "Young": 2.162,
    "Middle": 2.325,
    "Elder": 0.872,
    "Path_Change": 8.300,
    "Two_Pedestrians": 0.582,
    "More_Pedestrians": 0.135,
    "Three_Wheeler": 2.691,
    "Four_Wheeler": 1.615,
    "Heavy_Vehicle": 1.531,
    "Vehicle_Speed": 1.037,
    "Vehicle_Gap": 1.242,
    "Pedestrian_Speed": 1.343,
    "Waiting_Time": 1.121,
    "Lag": 2.054,
    "Vehicle_Encountered": 0.968,
}

# These are EXPLICIT ASSUMPTIONS because full distributions/means
# were not supplied in the information available from the paper.
CONTINUOUS_PARAMETERS = {
    "Vehicle_Speed": {
        "mean": 20.0, "sd": 5.63, "low": 5.05, "high": 36.94,
        "unit": "km/h"
    },
    "Vehicle_Gap": {
        "mean": 3.5, "sd": 1.24, "low": 1.01, "high": 7.0,
        "unit": "s"
    },
    "Waiting_Time": {
        "mean": 10.0, "sd": 7.52, "low": 1.01, "high": 43.0,
        "unit": "s"
    },
    "Pedestrian_Speed": {
        "mean": 1.20, "sd": 0.48, "low": 0.47, "high": 3.51,
        "unit": "m/s"
    },
    "Vehicle_Encountered": {
        "mean": 5.0, "sd": 5.01, "low": 0.0, "high": 26.0,
        "unit": "vehicles"
    },
}

# ============================================================
# 2. SAFE PROBABILITY NORMALIZATION
# ============================================================

def normalized_categories_and_probs(prob_dict):
    """
    Prevents the exact error encountered in the original notebook:
    NumPy requires probabilities to sum to exactly 1.

    Example:
    0.1963 + 0.8036 = 0.9999 due to rounded published percentages.
    We normalize them to sum exactly to 1.0.

    This is an implementation correction only; it does not change
    the intended published proportions in any meaningful way.
    """
    categories = list(prob_dict.keys())
    probs = np.asarray(list(prob_dict.values()), dtype=float)

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probability values must be finite.")

    if np.any(probs < 0):
        raise ValueError("Probabilities cannot be negative.")

    total = probs.sum()

    if total <= 0:
        raise ValueError("Probability sum must be greater than zero.")

    probs = probs / total
    return categories, probs


# Check every published categorical distribution
print("PROBABILITY CHECK")
print("-" * 70)

for name, prob_dict in {
    "Gender": GENDER_PROBS,
    "Age": AGE_PROBS,
    "Group": GROUP_PROBS,
    "Vehicle": VEHICLE_PROBS,
    "Lag/Gap": LAG_GAP_PROBS,
}.items():
    original_sum = sum(prob_dict.values())
    categories, probs = normalized_categories_and_probs(prob_dict)
    print(
        f"{name:12s} | original sum = {original_sum:.6f} "
        f"| normalized sum = {probs.sum():.6f}"
    )

print("\nThe gender values sum to 0.9999 because the paper reports rounded percentages.")
print("The normalization step fixes the NumPy choice() error.\n")


# ============================================================
# 3. BETA vs REPORTED ODDS RATIO CHECK
# ============================================================

or_check = []

for variable, beta in BETA.items():
    mathematical_or = np.exp(beta)
    reported_or = REPORTED_OR[variable]

    or_check.append({
        "Variable": variable,
        "Beta": beta,
        "exp(Beta)": mathematical_or,
        "Reported_OR": reported_or,
        "Difference": mathematical_or - reported_or,
    })

or_check_df = pd.DataFrame(or_check)

print("BETA vs REPORTED ODDS RATIO CHECK")
print("-" * 70)
print(or_check_df.to_string(index=False))

print("\nIMPORTANT MODEL INCONSISTENCY")
print(
    "Elder: beta = +0.120 gives exp(beta) = %.3f, "
    "but the paper reports OR = 0.872." % np.exp(0.120)
)
print(
    "For the mathematical simulation, the beta coefficient "
    "(+0.120) is used because logistic prediction is based on beta."
)
print(
    "The inconsistency should be reported as a limitation/possible "
    "table error in any presentation."
)


# ============================================================
# 4. TRUNCATED NORMAL SAMPLER
# ============================================================

def truncated_normal(size, mean, sd, low, high, rng):
    """
    Generate a truncated normal sample.

    IMPORTANT:
    The supplied paper does not provide complete distributions for
    all continuous variables. Therefore this is an explicit modeling
    assumption for a sensitivity/simulation exercise.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be positive.")
    if low >= high:
        raise ValueError("Lower bound must be smaller than upper bound.")

    a = (low - mean) / sd
    b = (high - mean) / sd

    return truncnorm.rvs(
        a,
        b,
        loc=mean,
        scale=sd,
        size=size,
        random_state=rng,
    )


# ============================================================
# 5. SYNTHETIC POPULATION GENERATOR
# ============================================================

def generate_synthetic_population(
    n=N,
    seed=42,
    three_wheeler_share=None,
):
    """
    Generate a synthetic population matching reported marginal
    distributions.

    This is NOT the original dataset.
    """

    rng = np.random.default_rng(seed)

    # Gender
    categories, probs = normalized_categories_and_probs(GENDER_PROBS)
    genders = rng.choice(categories, size=n, p=probs)

    # Age
    categories, probs = normalized_categories_and_probs(AGE_PROBS)
    ages = rng.choice(categories, size=n, p=probs)

    # Pedestrian group
    categories, probs = normalized_categories_and_probs(GROUP_PROBS)
    groups = rng.choice(categories, size=n, p=probs)

    # Vehicle type
    if three_wheeler_share is None:
        categories, probs = normalized_categories_and_probs(VEHICLE_PROBS)
    else:
        if not (0 <= three_wheeler_share <= 1):
            raise ValueError("three_wheeler_share must be between 0 and 1.")

        other = {
            k: v
            for k, v in VEHICLE_PROBS.items()
            if k != "Three-wheeler"
        }

        other_sum = sum(other.values())

        scenario_probs = {
            "Three-wheeler": three_wheeler_share
        }

        scenario_probs.update({
            k: (v / other_sum) * (1 - three_wheeler_share)
            for k, v in other.items()
        })

        categories, probs = normalized_categories_and_probs(scenario_probs)

    vehicles = rng.choice(
        categories,
        size=n,
        p=probs
    )

    # Lag / gap
    categories, probs = normalized_categories_and_probs(LAG_GAP_PROBS)
    lag_gap = rng.choice(categories, size=n, p=probs)

    # Continuous variables
    p = CONTINUOUS_PARAMETERS

    vehicle_speed = truncated_normal(
        n, p["Vehicle_Speed"]["mean"], p["Vehicle_Speed"]["sd"],
        p["Vehicle_Speed"]["low"], p["Vehicle_Speed"]["high"], rng
    )

    vehicle_gap = truncated_normal(
        n, p["Vehicle_Gap"]["mean"], p["Vehicle_Gap"]["sd"],
        p["Vehicle_Gap"]["low"], p["Vehicle_Gap"]["high"], rng
    )

    waiting_time = truncated_normal(
        n, p["Waiting_Time"]["mean"], p["Waiting_Time"]["sd"],
        p["Waiting_Time"]["low"], p["Waiting_Time"]["high"], rng
    )

    pedestrian_speed = truncated_normal(
        n, p["Pedestrian_Speed"]["mean"], p["Pedestrian_Speed"]["sd"],
        p["Pedestrian_Speed"]["low"], p["Pedestrian_Speed"]["high"], rng
    )

    vehicle_encountered = np.rint(
        truncated_normal(
            n,
            p["Vehicle_Encountered"]["mean"],
            p["Vehicle_Encountered"]["sd"],
            p["Vehicle_Encountered"]["low"],
            p["Vehicle_Encountered"]["high"],
            rng,
        )
    ).astype(int)

    # Path change is generated at reported prevalence.
    # It is NOT reconstructed from the original correlations.
    path_change = rng.binomial(
        1,
        TARGET_PATH_CHANGE_RATE,
        size=n
    )

    return pd.DataFrame({
        "Pedestrian_ID": np.arange(1, n + 1),
        "Gender": genders,
        "Age": ages,
        "Group": groups,
        "Vehicle_Type": vehicles,
        "Lag_Gap": lag_gap,
        "Vehicle_Speed": vehicle_speed,
        "Vehicle_Gap": vehicle_gap,
        "Waiting_Time": waiting_time,
        "Pedestrian_Speed": pedestrian_speed,
        "Vehicle_Encountered": vehicle_encountered,
        "Path_Change": path_change,
    })


population = generate_synthetic_population(
    n=N,
    seed=42
)

print("\nSynthetic population shape:", population.shape)
print("\nFirst five synthetic observations:")
print(population.head())


# ============================================================
# 6. CHECK THE SYNTHETIC MARGINAL DISTRIBUTIONS
# ============================================================

def percentage_table(series):
    return (
        series.value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("Percent")
        .to_frame()
    )

print("\nGENDER")
print(percentage_table(population["Gender"]))

print("\nAGE")
print(percentage_table(population["Age"]))

print("\nGROUP")
print(percentage_table(population["Group"]))

print("\nVEHICLE TYPE")
print(percentage_table(population["Vehicle_Type"]))

print("\nLAG/GAP")
print(percentage_table(population["Lag_Gap"]))

print("\nPATH CHANGE")
print(percentage_table(population["Path_Change"]))


# ============================================================
# 7. ENCODE THE PUBLISHED MODEL
# ============================================================

def encode_for_model(df):
    model_df = pd.DataFrame(index=df.index)

    # Gender
    model_df["Male"] = (df["Gender"] == "Male").astype(int)

    # Age; child is the reference
    model_df["Young"] = (df["Age"] == "Young").astype(int)
    model_df["Middle"] = (df["Age"] == "Middle").astype(int)
    model_df["Elder"] = (df["Age"] == "Elder").astype(int)

    # Path change
    model_df["Path_Change"] = df["Path_Change"]

    # Group; single is the reference
    model_df["Two_Pedestrians"] = (df["Group"] == "Two").astype(int)
    model_df["More_Pedestrians"] = (df["Group"] == "More").astype(int)

    # Vehicle type; two-wheeler is the reference
    model_df["Three_Wheeler"] = (
        df["Vehicle_Type"] == "Three-wheeler"
    ).astype(int)

    model_df["Four_Wheeler"] = (
        df["Vehicle_Type"] == "Four-wheeler"
    ).astype(int)

    model_df["Heavy_Vehicle"] = (
        df["Vehicle_Type"] == "Heavy"
    ).astype(int)

    # Continuous predictors
    model_df["Vehicle_Speed"] = df["Vehicle_Speed"]
    model_df["Vehicle_Gap"] = df["Vehicle_Gap"]
    model_df["Pedestrian_Speed"] = df["Pedestrian_Speed"]
    model_df["Waiting_Time"] = df["Waiting_Time"]

    # Gap reference; lag = 1
    model_df["Lag"] = (df["Lag_Gap"] == "Lag").astype(int)

    model_df["Vehicle_Encountered"] = df["Vehicle_Encountered"]

    return model_df


X = encode_for_model(population)


# ============================================================
# 8. LOGISTIC FUNCTIONS
# ============================================================

MODEL_TERMS = list(BETA.keys())


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def calculate_linear_predictor(X, intercept):
    z = np.full(len(X), intercept, dtype=float)

    for variable in MODEL_TERMS:
        z += BETA[variable] * X[variable].to_numpy()

    return z


def predicted_probability(X, intercept):
    z = calculate_linear_predictor(X, intercept)
    return sigmoid(z)


# ============================================================
# 9. CALIBRATE ILLUSTRATIVE INTERCEPT
# ============================================================

def intercept_objective(intercept):
    probabilities = predicted_probability(
        X,
        intercept
    )
    return probabilities.mean() - TARGET_SPEED_CHANGE_RATE


CALIBRATED_INTERCEPT = brentq(
    intercept_objective,
    -20,
    20
)

baseline_probability = predicted_probability(
    X,
    CALIBRATED_INTERCEPT
)

print("\nIllustrative calibrated intercept:")
print(round(CALIBRATED_INTERCEPT, 6))

print(
    "\nMean predicted baseline probability:",
    round(baseline_probability.mean(), 6)
)

print(
    "Published baseline prevalence:",
    TARGET_SPEED_CHANGE_RATE
)

print(
    "\nNOTE: This is an illustrative prevalence-calibrated intercept, "
    "NOT the original SPSS intercept."
)


# ============================================================
# 10. SIMULATED BASELINE OUTCOME
# ============================================================

rng = np.random.default_rng(12345)

population["Predicted_Probability"] = baseline_probability

population["Speed_Change_Simulated"] = rng.binomial(
    1,
    population["Predicted_Probability"]
)

simulated_rate = population[
    "Speed_Change_Simulated"
].mean()

print("\nSIMULATED BASELINE")
print("-" * 70)

print(
    "Simulated speed-change rate:",
    round(simulated_rate * 100, 2),
    "%"
)

print(
    "Published speed-change rate:",
    TARGET_SPEED_CHANGE_RATE * 100,
    "%"
)

print(
    "Number of simulated speed-change observations:",
    int(population["Speed_Change_Simulated"].sum()),
    "of",
    N
)


# ============================================================
# 11. ODDS-RATIO INTERPRETATIONS
# ============================================================

print("\nSELECTED ODDS-RATIO INTERPRETATIONS")
print("-" * 70)

speed_or_1 = np.exp(BETA["Vehicle_Speed"])
speed_or_5 = np.exp(BETA["Vehicle_Speed"] * 5)
speed_or_10 = np.exp(BETA["Vehicle_Speed"] * 10)

print(
    "Male vs Female odds multiplier:",
    round(np.exp(BETA["Male"]), 3)
)

print(
    "Three-wheeler vs two-wheeler odds multiplier:",
    round(np.exp(BETA["Three_Wheeler"]), 3)
)

print(
    "Lag vs Gap odds multiplier:",
    round(np.exp(BETA["Lag"]), 3)
)

print(
    "Per +1 km/h vehicle-speed odds multiplier:",
    round(speed_or_1, 3)
)

print(
    "Per +5 km/h vehicle-speed odds multiplier:",
    round(speed_or_5, 3)
)

print(
    "Per +10 km/h vehicle-speed odds multiplier:",
    round(speed_or_10, 3)
)


# ============================================================
# 12. SCENARIO FUNCTION
# ============================================================

def create_scenario_population(
    three_wheeler_share=None,
    vehicle_speed_reduction=0.0,
    seed=100
):
    scenario = generate_synthetic_population(
        n=N,
        seed=seed,
        three_wheeler_share=three_wheeler_share
    )

    if not (0 <= vehicle_speed_reduction < 1):
        raise ValueError(
            "vehicle_speed_reduction must be >= 0 and < 1."
        )

    scenario["Vehicle_Speed"] = (
        scenario["Vehicle_Speed"]
        * (1 - vehicle_speed_reduction)
    )

    scenario["Vehicle_Speed"] = np.clip(
        scenario["Vehicle_Speed"],
        CONTINUOUS_PARAMETERS["Vehicle_Speed"]["low"],
        CONTINUOUS_PARAMETERS["Vehicle_Speed"]["high"]
    )

    return scenario


def run_scenario(
    name,
    three_wheeler_share=None,
    vehicle_speed_reduction=0.0,
    seed=100
):
    scenario_pop = create_scenario_population(
        three_wheeler_share=three_wheeler_share,
        vehicle_speed_reduction=vehicle_speed_reduction,
        seed=seed
    )

    X_scenario = encode_for_model(
        scenario_pop
    )

    probabilities = predicted_probability(
        X_scenario,
        CALIBRATED_INTERCEPT
    )

    rng = np.random.default_rng(
        seed + 1000
    )

    outcomes = rng.binomial(
        1,
        probabilities
    )

    return {
        "Scenario": name,
        "3W_share_%": (
            three_wheeler_share * 100
            if three_wheeler_share is not None
            else np.nan
        ),
        "Vehicle_speed_reduction_%":
            vehicle_speed_reduction * 100,
        "Mean_predicted_probability":
            probabilities.mean(),
        "Simulated_speed_change_rate":
            outcomes.mean(),
        "Mean_vehicle_speed":
            scenario_pop["Vehicle_Speed"].mean(),
        "Observed_3W_share_in_synthetic_sample":
            (
                scenario_pop["Vehicle_Type"]
                .eq("Three-wheeler")
                .mean()
            ) * 100,
    }


# ============================================================
# 13. RUN SCENARIOS
# ============================================================

scenario_definitions = [
    ("Baseline", 0.3727, 0.00),
    ("10% speed reduction", 0.3727, 0.10),
    ("20% speed reduction", 0.3727, 0.20),
    ("30% speed reduction", 0.3727, 0.30),
    ("20% three-wheeler share", 0.20, 0.00),
    ("20% 3W + 20% speed reduction", 0.20, 0.20),
]

scenario_results = []

for i, (name, share, reduction) in enumerate(
    scenario_definitions
):
    scenario_results.append(
        run_scenario(
            name=name,
            three_wheeler_share=share,
            vehicle_speed_reduction=reduction,
            seed=100 + i
        )
    )

scenario_df = pd.DataFrame(
    scenario_results
)

baseline_prob = scenario_df.loc[
    scenario_df["Scenario"] == "Baseline",
    "Mean_predicted_probability"
].iloc[0]

scenario_df["Change_vs_Baseline_%"] = (
    scenario_df["Mean_predicted_probability"]
    / baseline_prob
    - 1
) * 100

print("\nSCENARIO RESULTS")
print("-" * 70)
print(
    scenario_df.round(4).to_string(index=False)
)


# ============================================================
# 14. FIGURE 1 — SCENARIO COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

x = np.arange(
    len(scenario_df)
)

plt.bar(
    x,
    scenario_df["Mean_predicted_probability"] * 100
)

plt.axhline(
    TARGET_SPEED_CHANGE_RATE * 100,
    linestyle="--",
    label="Published prevalence (30.85%)"
)

plt.xticks(
    x,
    scenario_df["Scenario"],
    rotation=35,
    ha="right"
)

plt.ylabel(
    "Mean predicted speed-change probability (%)"
)

plt.xlabel("Scenario")

plt.title(
    "Behavioral Speed-Change Propensity by Traffic Scenario"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "Figure_1_Scenario_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. FIGURE 2 — VEHICLE-SPEED RESPONSE
# ============================================================

speed_values = np.linspace(
    10,
    35,
    100
)

speed_probs = []

for speed in speed_values:

    temp = population.copy()

    temp["Vehicle_Speed"] = speed

    temp_X = encode_for_model(temp)

    probs = predicted_probability(
        temp_X,
        CALIBRATED_INTERCEPT
    )

    speed_probs.append(
        probs.mean() * 100
    )

plt.figure(figsize=(9, 6))

plt.plot(
    speed_values,
    speed_probs,
    linewidth=2
)

plt.xlabel(
    "Vehicle speed (km/h)"
)

plt.ylabel(
    "Mean predicted speed-change probability (%)"
)

plt.title(
    "Modeled Vehicle-Speed Response"
)

plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    "Figure_2_Vehicle_Speed_Response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. FIGURE 3 — THREE-WHEELER SHARE RESPONSE
# ============================================================

three_wheeler_values = np.array([
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.3727,
    0.40,
    0.45,
    0.50
])

three_wheeler_probs = []

for share in three_wheeler_values:

    temp = generate_synthetic_population(
        n=N,
        seed=500,
        three_wheeler_share=share
    )

    temp_X = encode_for_model(temp)

    probs = predicted_probability(
        temp_X,
        CALIBRATED_INTERCEPT
    )

    three_wheeler_probs.append(
        probs.mean() * 100
    )

plt.figure(figsize=(9, 6))

plt.plot(
    three_wheeler_values * 100,
    three_wheeler_probs,
    marker="o",
    linewidth=2
)

plt.axvline(
    37.27,
    linestyle="--",
    label="Published share = 37.27%"
)

plt.xlabel(
    "Three-wheeler share (%)"
)

plt.ylabel(
    "Mean predicted speed-change probability (%)"
)

plt.title(
    "Modeled Three-Wheeler-Share Response"
)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Figure_3_Three_Wheeler_Response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. RESPONSE SURFACE
# ============================================================

speed_grid = np.array([
    15, 20, 25, 30, 35
])

share_grid = np.array([
    0.10,
    0.20,
    0.30,
    0.3727,
    0.45
])

response_surface = np.zeros(
    (
        len(share_grid),
        len(speed_grid)
    )
)

for i, share in enumerate(share_grid):

    for j, speed in enumerate(speed_grid):

        temp = generate_synthetic_population(
            n=N,
            seed=600 + i * 10 + j,
            three_wheeler_share=share
        )

        temp["Vehicle_Speed"] = speed

        temp_X = encode_for_model(temp)

        probs = predicted_probability(
            temp_X,
            CALIBRATED_INTERCEPT
        )

        response_surface[i, j] = (
            probs.mean() * 100
        )

response_surface_df = pd.DataFrame(
    response_surface,
    index=[
        f"{share*100:.1f}%"
        for share in share_grid
    ],
    columns=[
        f"{speed:.0f}"
        for speed in speed_grid
    ]
)

print("\nBEHAVIORAL RESPONSE SURFACE (%)")
print(response_surface_df.round(2).to_string())


# Heatmap with matplotlib only
plt.figure(figsize=(9, 6))

plt.imshow(
    response_surface_df.values,
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    label="Predicted speed-change probability (%)"
)

plt.xticks(
    range(len(speed_grid)),
    [str(x) for x in speed_grid]
)

plt.yticks(
    range(len(share_grid)),
    [f"{x*100:.1f}%" for x in share_grid]
)

plt.xlabel(
    "Vehicle speed (km/h)"
)

plt.ylabel(
    "Three-wheeler share"
)

plt.title(
    "Behavioral Response Surface"
)

for i in range(response_surface_df.shape[0]):
    for j in range(response_surface_df.shape[1]):
        plt.text(
            j,
            i,
            f"{response_surface_df.iloc[i, j]:.1f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "Figure_4_Behavioral_Response_Surface.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. MONTE CARLO SCENARIO SIMULATION
# ============================================================

def monte_carlo_scenario(
    scenario_name,
    three_wheeler_share,
    vehicle_speed_reduction,
    repetitions=500,
    base_seed=10000
):
    rates = []

    for r in range(repetitions):

        scenario_pop = generate_synthetic_population(
            n=N,
            seed=base_seed + r,
            three_wheeler_share=three_wheeler_share
        )

        scenario_pop["Vehicle_Speed"] = np.clip(
            scenario_pop["Vehicle_Speed"]
            * (1 - vehicle_speed_reduction),
            CONTINUOUS_PARAMETERS["Vehicle_Speed"]["low"],
            CONTINUOUS_PARAMETERS["Vehicle_Speed"]["high"]
        )

        scenario_X = encode_for_model(
            scenario_pop
        )

        probs = predicted_probability(
            scenario_X,
            CALIBRATED_INTERCEPT
        )

        rng = np.random.default_rng(
            base_seed + 100000 + r
        )

        outcomes = rng.binomial(
            1,
            probs
        )

        rates.append(
            outcomes.mean()
        )

    rates = np.asarray(rates)

    return {
        "Scenario": scenario_name,
        "Mean_rate": rates.mean(),
        "SD": rates.std(ddof=1),
        "Lower_95": np.percentile(rates, 2.5),
        "Upper_95": np.percentile(rates, 97.5),
        "Min": rates.min(),
        "Max": rates.max(),
    }


mc_definitions = [
    ("Baseline", 0.3727, 0.00),
    ("10% speed reduction", 0.3727, 0.10),
    ("20% speed reduction", 0.3727, 0.20),
    ("30% speed reduction", 0.3727, 0.30),
    ("20% 3W share", 0.20, 0.00),
    ("20% 3W + 20% speed reduction", 0.20, 0.20),
]

mc_results = []

for i, (name, share, reduction) in enumerate(mc_definitions):

    mc_results.append(
        monte_carlo_scenario(
            scenario_name=name,
            three_wheeler_share=share,
            vehicle_speed_reduction=reduction,
            repetitions=500,
            base_seed=10000 + i * 5000
        )
    )

mc_df = pd.DataFrame(
    mc_results
)

print("\nMONTE CARLO RESULTS")
print("-" * 70)
print(mc_df.round(4).to_string(index=False))


# ============================================================
# 19. MONTE CARLO FIGURE
# ============================================================

plt.figure(figsize=(11, 6))

x = np.arange(
    len(mc_df)
)

means = mc_df["Mean_rate"].to_numpy() * 100
lower = mc_df["Lower_95"].to_numpy() * 100
upper = mc_df["Upper_95"].to_numpy() * 100

yerr = np.vstack([
    means - lower,
    upper - means
])

plt.errorbar(
    x,
    means,
    yerr=yerr,
    fmt="o",
    capsize=5
)

plt.axhline(
    TARGET_SPEED_CHANGE_RATE * 100,
    linestyle="--",
    label="Published prevalence"
)

plt.xticks(
    x,
    mc_df["Scenario"],
    rotation=35,
    ha="right"
)

plt.ylabel(
    "Simulated speed-change rate (%)"
)

plt.xlabel("Scenario")

plt.title(
    "Monte Carlo Behavioral Scenario Simulation"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "Figure_5_Monte_Carlo_Scenarios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. SENSITIVITY ANALYSIS
# ============================================================

sensitivity_variables = {
    "Vehicle_Speed": [10, 15, 20, 25, 30, 35],
    "Vehicle_Gap": [1.5, 2, 3, 4, 5, 6],
    "Waiting_Time": [2, 5, 10, 15, 20, 30, 40],
    "Pedestrian_Speed": [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    "Vehicle_Encountered": [0, 2, 5, 10, 15, 20, 25],
}

sensitivity_results = []

for variable, values in sensitivity_variables.items():

    for value in values:

        temp = population.copy()
        temp[variable] = value

        temp_X = encode_for_model(
            temp
        )

        probs = predicted_probability(
            temp_X,
            CALIBRATED_INTERCEPT
        )

        sensitivity_results.append({
            "Variable": variable,
            "Value": value,
            "Predicted_Probability":
                probs.mean()
        })

sensitivity_df = pd.DataFrame(
    sensitivity_results
)

print("\nSENSITIVITY RESULTS")
print(sensitivity_df.round(4).to_string(index=False))


# ============================================================
# 21. SENSITIVITY PLOTS
# ============================================================

for variable in sensitivity_variables:

    subset = sensitivity_df[
        sensitivity_df["Variable"] == variable
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        subset["Value"],
        subset["Predicted_Probability"] * 100,
        marker="o",
        linewidth=2
    )

    plt.xlabel(variable)
    plt.ylabel(
        "Mean predicted speed-change probability (%)"
    )

    plt.title(
        f"Sensitivity of Behavioral Propensity to {variable}"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    filename = (
        "Figure_Sensitivity_"
        + variable
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 22. SAVE RESULTS
# ============================================================

output_dir = Path(
    "pedestrian_behavior_simulation_outputs"
)

output_dir.mkdir(
    exist_ok=True
)

population.to_csv(
    output_dir / "synthetic_pedestrian_population.csv",
    index=False
)

or_check_df.to_csv(
    output_dir / "beta_or_check.csv",
    index=False
)

scenario_df.to_csv(
    output_dir / "scenario_results.csv",
    index=False
)

mc_df.to_csv(
    output_dir / "monte_carlo_results.csv",
    index=False
)

sensitivity_df.to_csv(
    output_dir / "sensitivity_results.csv",
    index=False
)

response_surface_df.to_csv(
    output_dir / "behavioral_response_surface.csv"
)

# Move generated figures
for filename in Path(".").glob("Figure_*.png"):
    target = output_dir / filename.name
    if target != filename:
        shutil.move(str(filename), str(target))

zip_path = shutil.make_archive(
    "pedestrian_behavior_simulation_outputs",
    "zip",
    output_dir
)

print("\nResults saved to:")
print(output_dir.resolve())

print("\nZIP archive:")
print(zip_path)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("Sample represented: 3,209 pedestrians")
print("Published speed-change prevalence: 30.85%")
print(
    "Illustrative calibrated intercept:",
    round(CALIBRATED_INTERCEPT, 4)
)
print(
    "Simulated baseline rate:",
    round(simulated_rate * 100, 2),
    "%"
)

print("\nPrimary outputs:")
print("1. Synthetic pedestrian population")
print("2. Published beta vs OR consistency check")
print("3. Logistic behavioral propensity")
print("4. Traffic-scenario comparison")
print("5. Vehicle-speed sensitivity")
print("6. Three-wheeler-share sensitivity")
print("7. Vehicle-speed x three-wheeler response surface")
print("8. Monte Carlo scenario intervals")

print("\nCRITICAL LIMITATIONS:")
print("- The original 3,209 observation-level dataset is unavailable.")
print("- The original logistic-model intercept is unavailable.")
print("- Several continuous-variable means/distributions are assumed.")
print("- Path-change is generated from its reported marginal prevalence.")
print("- This is a synthetic behavioral simulation, not PTV Vissim.")
print("- It is not a causal model and should not be presented as one.")
print("- It does not reproduce the original SPSS analysis.")

print("\nRecommended presentation wording:")
print(
    "A literature-derived stochastic behavioral simulation was developed "
    "using published logistic-regression coefficients and reported "
    "marginal distributions to examine counterfactual traffic scenarios."
)

print("=" * 80)
