import numpy as np
from datetime import datetime, timedelta

class CarbonProfile:
    def __init__(self, name, profile_type, base_intensity=300):
        self.name = name
        self.profile_type = profile_type
        self.base_intensity = base_intensity
    
    def get_intensity(self, timestamp):
        hour = (timestamp % 86400) / 3600
        
        if self.profile_type == "solar":
            if 6 <= hour <= 18:
                solar_factor = np.sin((hour - 6) * np.pi / 12)
                renewable_contribution = 0.7 * solar_factor
                return self.base_intensity * (1 - renewable_contribution)
            return self.base_intensity
        
        elif self.profile_type == "wind":
            wind_factor = 0.3 + 0.4 * np.sin(hour * np.pi / 12 + np.pi/4)
            return self.base_intensity * (1 - wind_factor)
        
        elif self.profile_type == "coal":
            return self.base_intensity * 2.5
        
        elif self.profile_type == "nuclear":
            return self.base_intensity * 0.3
        
        elif self.profile_type == "hydro":
            return self.base_intensity * 0.2
        
        elif self.profile_type == "mixed":
            daily_var = 0.15 * np.sin(hour * np.pi / 12)
            return self.base_intensity * (1 + daily_var)
        
        return self.base_intensity


class CarbonProfileManager:
    def __init__(self):
        self.profiles = {}
        self.node_assignments = {}
        self._create_default_profiles()
    
    def _create_default_profiles(self):
        self.profiles = {
            'solar_heavy': CarbonProfile('Solar Heavy', 'solar', base_intensity=250),
            'wind_heavy': CarbonProfile('Wind Heavy', 'wind', base_intensity=280),
            'coal_heavy': CarbonProfile('Coal Heavy', 'coal', base_intensity=400),
            'nuclear': CarbonProfile('Nuclear', 'nuclear', base_intensity=300),
            'hydro': CarbonProfile('Hydro', 'hydro', base_intensity=300),
            'mixed_grid': CarbonProfile('Mixed Grid', 'mixed', base_intensity=350)
        }
    
    def assign_profile(self, node_id, profile_name):
        if profile_name in self.profiles:
            self.node_assignments[node_id] = profile_name
        else:
            raise ValueError(f"Profile {profile_name} not found")
    
    def get_node_intensity(self, node_id, timestamp):
        if node_id not in self.node_assignments:
            profile_name = 'mixed_grid'
        else:
            profile_name = self.node_assignments[node_id]
        
        return self.profiles[profile_name].get_intensity(timestamp)
    
    def get_all_intensities(self, timestamp):
        intensities = {}
        for node_id in self.node_assignments:
            intensities[node_id] = self.get_node_intensity(node_id, timestamp)
        return intensities
    
    def assign_geographic_profiles(self, node_positions, num_nodes):
        for i in range(num_nodes):
            x, y = node_positions[i] if i in node_positions else (0.5, 0.5)
            
            if x < 0.3:
                profile = 'solar_heavy'
            elif x < 0.5:
                profile = 'wind_heavy'
            elif x < 0.7:
                profile = 'nuclear' if y > 0.5 else 'coal_heavy'
            else:
                profile = 'hydro' if y > 0.6 else 'mixed_grid'
            
            self.assign_profile(i, profile)
    
    def generate_time_series(self, duration_hours=24, interval_minutes=60):
        timestamps = np.arange(0, duration_hours * 3600, interval_minutes * 60)
        time_series = {}
        
        for node_id in self.node_assignments:
            intensities = [self.get_node_intensity(node_id, t) for t in timestamps]
            time_series[node_id] = {
                'timestamps': timestamps,
                'intensities': np.array(intensities)
            }
        
        return time_series


def create_realistic_profiles(num_nodes, topology_type='random'):
    manager = CarbonProfileManager()
    
    if topology_type == 'geographic':
        positions = {}
        for i in range(num_nodes):
            positions[i] = (np.random.random(), np.random.random())
        manager.assign_geographic_profiles(positions, num_nodes)
    
    elif topology_type == 'clustered':
        # Interleave high-carbon and low-carbon nodes throughout topology
        # This ensures routing decisions at ALL levels (core, agg, access) matter
        high_carbon = ['coal_heavy', 'mixed_grid']  # 800-1000 gCO2/kWh
        low_carbon = ['solar_heavy', 'wind_heavy', 'hydro', 'nuclear']  # 25-200 gCO2/kWh
        
        for i in range(num_nodes):
            if i % 3 == 0:
                # Every 3rd node is high-carbon (coal or mixed grid)
                profile = high_carbon[i % len(high_carbon)]
            else:
                # Other nodes are low-carbon
                profile = low_carbon[i % len(low_carbon)]
            manager.assign_profile(i, profile)
    
    else:
        profiles_list = list(manager.profiles.keys())
        for i in range(num_nodes):
            profile = np.random.choice(profiles_list)
            manager.assign_profile(i, profile)
    
    return manager


class CarbonIntensityManager:
    """Simpler carbon intensity manager for training data generation"""
    def __init__(self):
        self.node_profiles = {}
    
    def add_node_profile(self, node_id, hourly_intensities):
        """Add hourly carbon intensity profile for a node"""
        self.node_profiles[node_id] = hourly_intensities
    
    def get_node_intensity(self, node_id, timestamp):
        """Get carbon intensity for a node at given timestamp"""
        if node_id not in self.node_profiles:
            return 300  # Default
        
        hour = int((timestamp / 3600) % 24)
        return self.node_profiles[node_id][hour]


def create_multi_region_profiles(num_nodes, num_regions=3):
    """
    Create carbon profiles with high regional diversity for better training.
    
    This creates dramatic differences in carbon intensity across regions,
    enabling significant carbon savings through intelligent routing.
    
    Args:
        num_nodes: Total number of nodes
        num_regions: Number of regions (default 3)
        
    Returns:
        CarbonIntensityManager with diverse regional profiles
        
    Region characteristics:
    - Region 1 (clean): 50-150 gCO2/kWh (solar/wind heavy, e.g., Iceland, Norway)
    - Region 2 (medium): 200-400 gCO2/kWh (mixed grid, e.g., California, UK)
    - Region 3 (dirty): 500-800 gCO2/kWh (coal heavy, e.g., Poland, China)
    """
    manager = CarbonIntensityManager()
    
    # Assign nodes to regions
    nodes_per_region = num_nodes // num_regions
    
    for node_id in range(num_nodes):
        region = min(node_id // nodes_per_region, num_regions - 1)
        
        if region == 0:  # Clean energy region
            base_intensity = np.random.uniform(50, 150)
        elif region == 1:  # Medium carbon region
            base_intensity = np.random.uniform(200, 400)
        else:  # High carbon region
            base_intensity = np.random.uniform(500, 800)
        
        # Add realistic time-of-day variation (±30%)
        # Solar peaks at noon, wind more constant, coal/gas adjust for demand
        hourly_pattern = []
        for hour in range(24):
            # Sinusoidal pattern with peak at noon for solar
            time_factor = 1 + 0.3 * np.sin((hour - 6) * np.pi / 12)
            # Add some randomness
            time_factor *= np.random.uniform(0.95, 1.05)
            hourly_pattern.append(base_intensity * time_factor)
        
        manager.add_node_profile(node_id, hourly_pattern)
    
    return manager


if __name__ == "__main__":
    print("Carbon Profile Manager Test")
    print("=" * 50)
    
    manager = create_realistic_profiles(10, 'clustered')
    
    test_times = [0, 21600, 43200, 64800]
    time_labels = ['Midnight', '6 AM', 'Noon', '6 PM']
    
    for t, label in zip(test_times, time_labels):
        print(f"\n{label} (t={t}s):")
        intensities = manager.get_all_intensities(t)
        for node_id, intensity in intensities.items():
            profile_name = manager.node_assignments[node_id]
            print(f"  Node {node_id} ({profile_name}): {intensity:.1f} gCO2/kWh")
    
    print("\nCarbon profile test passed.")
