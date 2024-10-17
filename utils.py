import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# machine_statistics 프린트
def print_machine_statistics(model, total_simulation_time):
    print(total_simulation_time)
    total_utilization = 0
    utilization_rates = []
    machine_names = []
    total_waiting_times = []
    total_transportation_time = 0

    for machine in model.values():
        if isinstance(machine, Machine):
            total_operations = len(machine.workingtime_log)
            total_utilization_time = sum(machine.workingtime_log)
            utilization_rate = (
                                           total_utilization_time / total_simulation_time) * 100 if total_simulation_time > 0 else 0
            # machine.total_waiting_time = sum([op.wait_time for op in machine.queue])

            total_utilization += utilization_rate
            utilization_rates.append(utilization_rate)
            machine_names.append(machine.name)
            total_waiting_times.append(machine.waiting_time)
            total_working_time = sum(machine.workingtime_log)
            total_transportation_time += machine.transportation_time

            print(f"{machine.name} statistics:")
            print(f"  Total operations: {total_operations}")
            print(f"  Total waiting time: {machine.waiting_time:.2f} hours")
            print(f"  Total utilization time: {total_utilization_time:.2f} hours")
            print(f"  Utilization rate: {utilization_rate:.2f}%")
            print(f"  Working time log: {machine.workingtime_log}")
            print(f"  Total working time: {total_working_time}")

    average_utilization = total_utilization / len(utilization_rates) if utilization_rates else 0
    std_dev_utilization = np.std(utilization_rates)

    print(f"Total transportation time: {total_transportation_time} hours")
    print(f"Average utilization rate: {average_utilization:.2f}%")
    print(f"Standard deviation of utilization rate: {std_dev_utilization:.2f}%")
    print(f"Total waiting times : {int(sum(total_waiting_times))} hours")

    # # 그래프 생성
    # plt.figure(figsize=(10, 6))
    # plt.bar(machine_names, utilization_rates, color='blue')
    # plt.xlabel('Machine')
    # plt.ylabel('Utilization Rate (%)')
    # plt.title('Machine Utilization Rates')
    # plt.ylim(0, 100)

    # Plotting
    sns.set(style="whitegrid")
    palette = sns.color_palette("pastel", len(machine_names))

    fig, ax1 = plt.subplots()

    ax1.bar(machine_names, utilization_rates, color=palette)
    ax1.set_xlabel('Machines')
    ax1.set_ylabel('Utilization Rate (%)')
    ax1.set_title('Machine Utilization Rates')
    plt.show()