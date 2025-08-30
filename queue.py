#6.
import numpy as np
import matplotlib.pyplot as plt

total_simulation_time = 1000
mean_inter_arrival = 10
min_service_duration = 8
max_service_duration = 12

customer_arrival_times = []
simulation_clock = 0

while simulation_clock < total_simulation_time:
    time_between_arrivals = np.random.exponential(scale=mean_inter_arrival)
    simulation_clock += time_between_arrivals
    if simulation_clock < total_simulation_time:
        customer_arrival_times.append(simulation_clock)

print(f"Total customers arrived: {len(customer_arrival_times)} within {total_simulation_time} minutes.")

customer_service_durations = []

for _ in customer_arrival_times:
    duration = -1
    while not (min_service_duration <= duration <= max_service_duration):
        duration = np.random.poisson(lam=10)
    customer_service_durations.append(duration)

print(f"Total service durations generated: {len(customer_service_durations)}.")

service_start_times = []
service_end_times = []
queue_wait_times = []
server_occupied_until = 0

for i in range(len(customer_arrival_times)):
    customer_arrival = customer_arrival_times[i]
    service_duration = customer_service_durations[i]

    service_begin = max(customer_arrival, server_occupied_until)
    wait_duration = service_begin - customer_arrival
    service_complete = service_begin + service_duration

    service_start_times.append(service_begin)
    service_end_times.append(service_complete)
    queue_wait_times.append(wait_duration)

    server_occupied_until = service_complete

print(f"Total service start times calculated: {len(service_start_times)}.")
print(f"Total service end times calculated: {len(service_end_times)}.")
print(f"Total wait times calculated: {len(queue_wait_times)}.")

avg_queue_wait = sum(queue_wait_times) / len(queue_wait_times)

customer_arrival_rate = len(customer_arrival_times) / total_simulation_time
avg_customers_in_queue = customer_arrival_rate * avg_queue_wait

total_service_duration = sum(customer_service_durations)
server_usage_rate = total_service_duration / total_simulation_time

print(f"Average wait time per customer: {avg_queue_wait:.2f} minutes")
print(f"Average number of customers waiting in queue: {avg_customers_in_queue:.2f}")
print(f"Server utilization rate: {server_usage_rate:.2f}")

event_list = []
for arrival in customer_arrival_times:
    event_list.append((arrival, 1))

for completion in service_end_times:
    event_list.append((completion, -1))

event_list.sort(key=lambda x: x[0])

timeline = [0]
queue_sizes = [0]
current_queue_size = 0
current_timestamp = 0

for event_timestamp, event_change in event_list:
    if event_timestamp > current_timestamp:
        timeline.append(event_timestamp)
        queue_sizes.append(current_queue_size)

    current_timestamp = event_timestamp
    current_queue_size += event_change

    timeline.append(current_timestamp)
    queue_sizes.append(current_queue_size)

plt.figure(figsize=(12, 6))
plt.step(timeline, queue_sizes, where='post')
plt.xlabel('Simulation Time (minutes)')
plt.ylabel('Customers in Queue')
plt.title('Queue Size Over Simulation Time')
plt.grid(True)
plt.show()
