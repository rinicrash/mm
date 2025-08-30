import numpy as np

days = [200, 500, 1000, 10000]
purchase = 50
sell_price = 0.45
scrap_price = 0.05
news_types = {'Good': 0.35, 'Fair': 0.45, 'Poor': 0.20}
demand_cdf = [0.03, 0.10, 0.28, 0.68, 0.82, 0.94, 1.00]
demand_values = [40, 50, 60, 70, 80, 90, 100]

# For different number of days
for num_days in days:
    # Generate news types
    random_digits = np.random.uniform(0, 100, num_days)
    news = []
    for digit in random_digits:
        if digit < 35:
            news.append('Good')
        elif digit < 80:
            news.append('Fair')
        else:
            news.append('Poor')

    # Generate demand based on news type
    demands = []
    for news_type in news:
        if news_type == 'Good':
            demand_digit = np.random.exponential(scale=50)
        elif news_type == 'Fair':
            demand_digit = np.random.normal(50, 10)
        else:  # Poor
            demand_digit = np.random.uniform(0, 100)

        # Convert demand_digit to a probability value between 0 and 1
        prob = demand_digit / 100
        if prob > 1:
            prob = 1.0

        # Find which interval this probability falls into
        for i, cdf in enumerate(demand_cdf):
            if prob <= cdf:
                demand = demand_values[i]
                break

        demands.append(demand)


    revenues = []
    loss_of_profits = []
    salvages = []
    daily_profits = []
    for demand in demands:
        sold = min(purchase, demand)
        revenue = sold * sell_price  # (i) Revenue from sales
        revenues.append(revenue)
        excess_demand = max(0, demand - purchase)
        loss_profit = excess_demand * sell_price  # (ii) Loss of profit from excess demand
        loss_of_profits.append(loss_profit)
        unsold = max(0, purchase - demand)
        salvage = unsold * scrap_price  # (iii) Salvage from scrap
        salvages.append(salvage)
        profit = revenue + salvage - loss_profit  # (iv) Daily profit
        daily_profits.append(profit)


    print(f"\nResults for {num_days} days:")
    print("-" * 50)
    print(f"{'Day':<6} {'Revenue (i)':<12} {'Loss Profit (ii)':<16} {'Salvage (iii)':<12} {'Daily Profit (iv)':<12}")
    print("-" * 50)
    for i in range(min(num_days, 10)):
        print(f"{i+1:<6} {revenues[i]:<12.2f} {loss_of_profits[i]:<16.2f} {salvages[i]:<12.2f} {daily_profits[i]:<12.2f}")
    if num_days > 10:
        print(f"... (showing first 10 of {num_days} days)")
    print("-" * 50)
    # Print averages for all days
    print(f"Average Revenue (i): {np.mean(revenues):.2f}")
    print(f"Average Loss of Profit (ii): {np.mean(loss_of_profits):.2f}")
    print(f"Average Salvage (iii): {np.mean(salvages):.2f}")
    print(f"Average Daily Profit (iv): {np.mean(daily_profits):.2f}\n")
