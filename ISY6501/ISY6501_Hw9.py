import simpy
import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
ARRIVAL_RATE = 5  # Passengers per minute (Poisson)
ID_CHECK_TIME = 0.75  # Mean service time (Exponential)
SCANNER_TIME = (0.5, 1.0)  # Uniform distribution
SIM_TIME = 120  # Simulating 2 hours

class AirportSecurity:
    def __init__(self, env, num_id_checkers, num_scanners):
        self.env = env
        self.id_check = simpy.Resource(env, num_id_checkers)
        self.scanner = simpy.Resource(env, num_scanners)

    def check_id(self, passenger):
        yield self.env.timeout(random.expovariate(1.0 / ID_CHECK_TIME))

    def scan_passenger(self, passenger):
        yield self.env.timeout(random.uniform(*SCANNER_TIME))

def passenger(env, name, airport):
    arrival_time = env.now
    with airport.id_check.request() as request:
        yield request
        yield env.process(airport.check_id(name))

    with airport.scanner.request() as request:
        yield request
        yield env.process(airport.scan_passenger(name))

    total_time = env.now - arrival_time
    wait_times.append(total_time)

def run_simulation(num_id_checkers, num_scanners):
    global wait_times
    wait_times = []

    env = simpy.Environment()
    airport = AirportSecurity(env, num_id_checkers, num_scanners)

    for i in range(ARRIVAL_RATE * SIM_TIME):
        env.process(passenger(env, f"Passenger-{i}", airport))
        env.timeout(random.expovariate(1.0 / ARRIVAL_RATE))  # Remove yield

    env.run(until=SIM_TIME)
    
    return np.mean(wait_times)  # Ensure a numeric return value

# Running simulations with different numbers of ID checkers and scanners
results = {}
for id_checkers in range(2, 6):
    for scanners in range(2, 6):
        avg_wait = run_simulation(id_checkers, scanners)
        results[(id_checkers, scanners)] = avg_wait

# Find the optimal configuration
optimal_config = min(results, key=results.get)
print(f"Optimal number of ID Checkers: {optimal_config[0]}, Scanners: {optimal_config[1]}")


df = pd.DataFrame(results.items(), columns=["Config", "Avg Wait Time"])
df[["ID Checkers", "Scanners"]] = pd.DataFrame(df["Config"].tolist(), index=df.index)

sns.heatmap(df.pivot("ID Checkers", "Scanners", "Avg Wait Time"), annot=True, cmap="coolwarm")
plt.title("Average Wait Time for Different Configurations")
plt.show()