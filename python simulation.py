"""
MIMO-RLS-LQR Adaptive Governor Simulation
Author: Stanislav Usychenko
"""

import numpy as np
import matplotlib.pyplot as plt
from governor import MIMO_RLS_LQR_Governor


def run_simulation(steps: int = 1000, seed: int = 42):
    """Run full simulation of the adaptive governor"""
    np.random.seed(seed)
    governor = MIMO_RLS_LQR_Governor(lambda_forget=0.975)
    
    # True system model (for simulation only)
    A_true = np.array([
        [0.92, 0.05, 0.0, 0.1,  0.0,  0.0],
        [0.08, 0.88, 0.1,  0.0,  0.0,  0.0],
        [0.0,  0.12, 0.85, 0.0,  0.05, 0.1],
        [0.15, 0.0,  0.0,  0.9,  0.0,  0.0],
        [0.0,  0.0,  0.08, 0.0,  0.95, 0.0],
        [0.0,  0.05, 0.12, 0.0,  0.0,  0.88]
    ])
    
    B_true = np.array([
        [0.3,  0.1],
        [0.2,  0.25],
        [0.15, 0.2],
        [0.4,  0.05],
        [0.1,  0.15],
        [0.05, 0.3]
    ])
    
    x = np.zeros(6)
    history = {
        'x': [],
        'u': [],
        'temp': [],
        'energy': []
    }
    
    for k in range(steps):
        history['x'].append(x.copy())
        history['temp'].append(x[0])                    # Tnorm
        
        # Get control action
        u = governor.get_control(x)
        history['u'].append(u.copy())
        
        # Simulate real system dynamics + noise
        x_next = A_true @ x + B_true @ u + np.random.normal(0, 0.015, 6)
        x_next = np.clip(x_next, 0.0, 1.2)
        
        # Online system identification (MIMO-RLS)
        governor.update_rls(x, x_next, u)
        
        # Periodic LQR gain update
        if k % 25 == 0:
            A_est, B_est = governor.extract_ab()
            try:
                governor.K = governor.compute_lqr_gain(A_est, B_est)
            except:
                pass  # Keep previous K if computation fails
        
        x = x_next
        history['energy'].append(np.sum(np.abs(u)))
    
    return history


if __name__ == "__main__":
    print("🚀 Running MIMO-RLS-LQR Adaptive Governor Simulation...\n")
    
    data = run_simulation(steps=1000)
    
    # Results
    print("=== SIMULATION RESULTS ===")
    print(f"Average Temperature (Tnorm): {np.mean(data['temp']):.3f}")
    print(f"Maximum Temperature (Tnorm): {np.max(data['temp']):.3f}")
    print(f"Total Control Energy:        {np.sum(data['energy']):.2f}")
    print(f"Simulation steps:            {len(data['temp'])}")
    
    # Plotting
    plt.figure(figsize=(12, 9))
    
    plt.subplot(3, 1, 1)
    plt.plot(data['temp'], 'b-', label='Tnorm')
    plt.title('Temperature (Normalized)')
    plt.ylabel('Tnorm')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    u_array = np.array(data['u'])
    plt.plot(u_array[:, 0], 'b-', label='u_big')
    plt.plot(u_array[:, 1], 'orange', label='u_little')
    plt.title('Control Inputs (Δ Frequency)')
    plt.ylabel('Control Action')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(np.array(data['x'])[:, 2], 'g-', label='Lnorm')
    plt.title('System Load (Normalized)')
    plt.xlabel('Time Step')
    plt.ylabel('Lnorm')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=200, bbox_inches='tight')
    print("\n✅ Plot saved as 'simulation_results.png'")
    plt.show()