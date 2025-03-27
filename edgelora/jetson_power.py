from jtop import jtop
import time
import numpy as np

def measure_average_power(duration_seconds=60):
    power_measurements = []
    
    with jtop() as jetson:
        start_time = time.time()
        
        while jetson.ok():
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Get total power
            if jetson.power:
                total_power = jetson.power['tot']['power']
                power_measurements.append(total_power)
                print(f"Current power: {total_power:.2f}mW")
            
            # Check if we've reached the duration
            if elapsed_time >= duration_seconds:
                break
            
            # Sleep briefly to avoid overwhelming the system
            time.sleep(0.5)
    
    # Calculate statistics
    avg_power = np.mean(power_measurements)
    std_power = np.std(power_measurements)
    min_power = np.min(power_measurements)
    max_power = np.max(power_measurements)
    
    print("\nPower Statistics:")
    print(f"Average Power: {avg_power:.2f}mW")
    print(f"Std Dev: {std_power:.2f}mW")
    print(f"Min Power: {min_power:.2f}mW")
    print(f"Max Power: {max_power:.2f}mW")
    print(f"Number of samples: {len(power_measurements)}")
    
    return avg_power

if __name__ == "__main__":
    print("Measuring average power for 60 seconds...")
    measure_average_power()