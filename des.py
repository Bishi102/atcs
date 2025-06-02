import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.spatial import Delaunay


class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time


class CoffeeShopSimulator:
    def __init__(self,
                 num_baristas=2,
                 coffee_price=5.0,
                 wage_per_hour=20.0,
                 shop_start_time=6.0,
                 shop_open_hours=12,
                 max_queue_length=5):
        self.num_baristas = num_baristas
        self.coffee_price = coffee_price
        self.wage_per_hour = wage_per_hour
        self.start_time = shop_start_time
        self.shop_open_hours = shop_open_hours
        self.max_queue_length = max_queue_length

        self.customer_penalty = coffee_price * 0.3
        self.service_time_mean = self._compute_service_time_mean()
        self.base_price = 2.0
        self.price_elasticity = -0.2

        self.queue = []
        self.busy_baristas = 0
        self.total_revenue = 0
        self.total_penalty = 0

    def _compute_service_time_mean(self):
        min_time = 1.5 / 60
        max_time = 5.0 / 60
        base_wage = 20.0
        k = 0.4

        wage_diff = max(self.wage_per_hour - base_wage, 0)
        return min_time + (max_time - min_time) * math.exp(-k * wage_diff)

    def _gaussian(self, x, mu, sigma, scale):
        return scale * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def _time_dependent_rate(self, t_hour):
        base_rate = self._gaussian(t_hour, mu=7, sigma=0.7, scale=350) + \
                    self._gaussian(t_hour, mu=9, sigma=0.7, scale=330)
        price_factor = (self.coffee_price / self.base_price) ** self.price_elasticity
        return base_rate * price_factor

    def _get_next_arrival(self, current_time, step_size=1/60):
        t = current_time + step_size
        end_time = self.start_time + self.shop_open_hours
        while t <= end_time:
            rate = self._time_dependent_rate(t)
            prob = rate * step_size
            if random.random() < prob:
                return t
            t += step_size
        return float('inf')

    def _get_service_time(self):
        return random.expovariate(1 / self.service_time_mean)

    def run(self):
        current_time = self.start_time
        end_time = self.start_time + self.shop_open_hours
        event_queue = []
        heapq.heappush(event_queue, Event(self._get_next_arrival(current_time), 'arrival'))

        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time
            if current_time > end_time:
                break

            if event.event_type == 'arrival':
                next_arrival = self._get_next_arrival(current_time)
                if next_arrival <= end_time:
                    heapq.heappush(event_queue, Event(next_arrival, 'arrival'))

                if self.busy_baristas < self.num_baristas:
                    self.busy_baristas += 1
                    service_time = self._get_service_time()
                    heapq.heappush(event_queue, Event(current_time + service_time, 'departure'))
                    self.total_revenue += self.coffee_price
                elif len(self.queue) < self.max_queue_length:
                    self.queue.append(current_time)
                else:
                    self.total_penalty += self.customer_penalty

            elif event.event_type == 'departure':
                if self.queue:
                    self.queue.pop(0)
                    service_time = self._get_service_time()
                    heapq.heappush(event_queue, Event(current_time + service_time, 'departure'))
                    self.total_revenue += self.coffee_price
                else:
                    self.busy_baristas -= 1

        total_wages = self.num_baristas * self.wage_per_hour * self.shop_open_hours
        profit = self.total_revenue - total_wages - self.total_penalty

        return {
            'profit': profit,
            'revenue': self.total_revenue,
            'wages': total_wages,
            'penalties': self.total_penalty
        }


# Globals to track progress (reset before each run)
iteration_history = []
best_so_far_history = []

def objective_function(params):
    coffee_price, num_baristas = params
    num_baristas = int(round(num_baristas))

    # Run simulation 365 times (simulate a year) and average profit
    profits = []
    for _ in range(365):
        sim = CoffeeShopSimulator(
            coffee_price=coffee_price,
            num_baristas=num_baristas,
            max_queue_length=5  # fixed max queue length
        )
        result = sim.run()
        profits.append(result['profit'])
    avg_profit = np.mean(profits)

    # Track iteration results
    iteration_history.append(avg_profit)
    current_best = max(best_so_far_history) if best_so_far_history else -np.inf
    if avg_profit > current_best:
        best_so_far_history.append(avg_profit)
    else:
        best_so_far_history.append(current_best)

    return -avg_profit  # We minimize negative profit to maximize profit


bounds = [
    (2.5, 8.0),    # coffee_price
    (1, 5),        # num_baristas
    (18, 25),      # wage_per_hour
    (6.0, 6.0),    # shop_start_time fixed to 6am
    (6, 14)        # shop_open_hours
]

def run_optimization(seed):
    np.random.seed(seed)
    random.seed(seed)
    iteration_history.clear()
    best_so_far_history.clear()

    result = differential_evolution(objective_function, bounds, maxiter=100, disp=False)
    optimized_profit = -result.fun
    return optimized_profit



if __name__ == "__main__":
    np.random.seed(69)
    
    optimized_profits = []
    runs = 20

    for run_i in range(runs):
        print(f"Starting optimization run {run_i+1}/{runs} ...")
        profit = run_optimization(run_i)
        print(f"Run {run_i+1} optimized profit: ${profit:.2f}")
        optimized_profits.append(profit)

    mean_profit = np.mean(optimized_profits)
    std_profit = np.std(optimized_profits)

    print(f"\nAverage optimized profit over {runs} runs: ${mean_profit:.2f}")
    print(f"Standard deviation of optimized profit over {runs} runs: ${std_profit:.2f}")

    number_of_points = 10
    initial_points = sample_initial_points(number_of_points, bounds)
    triangulation = deluanay_triangulation(initial_points)

    print("Initial sampled points:")
    print(initial_points)
    print("\nDelaunay simplices (indices of points forming each simplex):")
    print(triangulation.simplices)
