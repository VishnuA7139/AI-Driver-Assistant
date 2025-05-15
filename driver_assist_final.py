import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

# Load dataset
df = pd.read_csv("driver_assist_dataset.csv")

X = df[['distance', 'traffic', 'road_condition']].values
y = df['safe_speed'].values

# Train neural network
nn_model = MLPRegressor(hidden_layer_sizes=(15, 10), max_iter=2000, random_state=42)
nn_model.fit(X, y)

# Fuzzy logic system
distance = ctrl.Antecedent(np.arange(0, 51, 1), 'distance')
road_quality = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'road_quality')
speed = ctrl.Consequent(np.arange(0, 101, 1), 'speed')

distance['near'] = fuzz.trimf(distance.universe, [0, 0, 15])
distance['medium'] = fuzz.trimf(distance.universe, [10, 25, 40])
distance['far'] = fuzz.trimf(distance.universe, [30, 50, 50])

road_quality['poor'] = fuzz.trimf(road_quality.universe, [0, 0, 0.4])
road_quality['average'] = fuzz.trimf(road_quality.universe, [0.3, 0.5, 0.7])
road_quality['good'] = fuzz.trimf(road_quality.universe, [0.6, 1, 1])

speed['very_slow'] = fuzz.trimf(speed.universe, [0, 0, 30])
speed['slow'] = fuzz.trimf(speed.universe, [20, 35, 50])
speed['moderate'] = fuzz.trimf(speed.universe, [40, 55, 70])
speed['fast'] = fuzz.trimf(speed.universe, [60, 80, 100])

rules = [
    ctrl.Rule(distance['near'] & road_quality['poor'], speed['very_slow']),
    ctrl.Rule(distance['near'] & road_quality['average'], speed['slow']),
    ctrl.Rule(distance['medium'] & road_quality['average'], speed['moderate']),
    ctrl.Rule(distance['medium'] & road_quality['good'], speed['moderate']),
    ctrl.Rule(distance['far'] & road_quality['good'], speed['fast'])
]

fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

test_cases = [
    [12, 2, 0.6],
    [5, 5, 0.2],
    [30, 1, 0.9],
    [20, 3, 0.5],
    [15, 2, 0.7]
]

os.makedirs("simulation_outputs", exist_ok=True)

for i, case in enumerate(test_cases):
    dist, traffic, road = case
    nn_speed = nn_model.predict(np.array(case).reshape(1, -1))[0]
    fuzzy_sim.input['distance'] = dist
    fuzzy_sim.input['road_quality'] = road
    fuzzy_sim.compute()
    fuzzy_speed = fuzzy_sim.output['speed']
    plt.figure(figsize=(5, 4))
    plt.bar(['Data-Driven Model', 'Rule-Based Estimator'], [nn_speed, fuzzy_speed],
            color=['skyblue', 'darkorange'])
    plt.ylabel("Predicted Speed (km/h)")
    plt.title(f"CASE {i+1}: Predicted Speeds")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"simulation_outputs/case_{i+1}_output.png")
    plt.close()

