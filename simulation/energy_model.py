import numpy as np

class EnergyModel:
    def __init__(self, node_id, energy_type='grid', initial_energy=1e9):
        self.node_id = node_id
        self.energy_type = energy_type
        self.initial_energy = initial_energy
        self.remaining_energy = initial_energy
        
        self.idle_power = 50
        self.tx_power = 150
        self.rx_power = 100
        
        self.tx_current = 0.300
        self.rx_current = 0.200
        self.idle_current = 0.100
        
        self.voltage = 5.0
        
        self.state = 'IDLE'
        self.tx_packets = 0
        self.rx_packets = 0
        self.total_tx_time = 0
        self.total_rx_time = 0
    
    def update_state(self, new_state, duration=1.0):
        energy_consumed = self._calculate_energy(self.state, duration)
        self.remaining_energy -= energy_consumed
        
        if new_state == 'TX':
            self.tx_packets += 1
            self.total_tx_time += duration
        elif new_state == 'RX':
            self.rx_packets += 1
            self.total_rx_time += duration
        
        self.state = new_state
        return energy_consumed
    
    def _calculate_energy(self, state, duration):
        if state == 'TX':
            power = self.tx_power
        elif state == 'RX':
            power = self.rx_power
        else:
            power = self.idle_power
        
        return power * duration
    
    def get_power_consumption(self):
        if self.state == 'TX':
            return self.tx_power
        elif self.state == 'RX':
            return self.rx_power
        return self.idle_power
    
    def get_energy_ratio(self):
        if self.initial_energy > 0:
            return self.remaining_energy / self.initial_energy
        return 1.0
    
    def process_packet(self, packet_size, is_tx=True):
        bit_energy = 1e-9
        energy_for_packet = packet_size * 8 * bit_energy
        
        if is_tx:
            base_energy = self.tx_power * 0.001
        else:
            base_energy = self.rx_power * 0.001
        
        total_energy = base_energy + energy_for_packet
        self.remaining_energy -= total_energy
        
        if is_tx:
            self.tx_packets += 1
        else:
            self.rx_packets += 1
        
        return total_energy


class NetworkEnergyManager:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.energy_models = {}
        self.total_energy_consumed = 0
        self.energy_history = []
        
    def initialize_nodes(self, node_types=None):
        if node_types is None:
            node_types = ['grid'] * self.num_nodes
        
        for i in range(self.num_nodes):
            energy_type = node_types[i] if i < len(node_types) else 'grid'
            
            if energy_type == 'battery':
                initial = 1e6
            elif energy_type == 'solar':
                initial = 5e6
            else:
                initial = 1e9
            
            self.energy_models[i] = EnergyModel(i, energy_type, initial)
    
    def update_node_state(self, node_id, state, duration=1.0):
        if node_id in self.energy_models:
            energy = self.energy_models[node_id].update_state(state, duration)
            self.total_energy_consumed += energy
            return energy
        return 0
    
    def process_transmission(self, src_node, dst_node, packet_size):
        energy_tx = 0
        energy_rx = 0
        
        if src_node in self.energy_models:
            energy_tx = self.energy_models[src_node].process_packet(packet_size, is_tx=True)
        
        if dst_node in self.energy_models:
            energy_rx = self.energy_models[dst_node].process_packet(packet_size, is_tx=False)
        
        total = energy_tx + energy_rx
        self.total_energy_consumed += total
        return total
    
    def get_node_energy_ratio(self, node_id):
        if node_id in self.energy_models:
            return self.energy_models[node_id].get_energy_ratio()
        return 1.0
    
    def get_all_energy_ratios(self):
        return {nid: model.get_energy_ratio() for nid, model in self.energy_models.items()}
    
    def get_total_power(self):
        return sum(model.get_power_consumption() for model in self.energy_models.values())
    
    def calculate_carbon_emission(self, carbon_intensities, duration=1.0):
        total_carbon = 0
        
        for node_id, model in self.energy_models.items():
            power_watts = model.get_power_consumption()
            energy_kwh = (power_watts / 1000.0) * (duration / 3600.0)
            
            carbon_intensity = carbon_intensities.get(node_id, 350)
            carbon_emission = energy_kwh * carbon_intensity
            total_carbon += carbon_emission
        
        return total_carbon
    
    def calculate_carbon_with_traffic(self, carbon_intensities, node_traffic_loads, duration=1.0):
        """
        Calculate carbon emissions based on actual traffic loads
        
        Args:
            carbon_intensities: Dict mapping node_id to carbon intensity (gCO2/kWh)
            node_traffic_loads: Dict mapping node_id to traffic load (Gbps)
            duration: Time period in seconds
        
        Returns:
            Total carbon emissions in gCO2
        """
        total_carbon = 0
        
        for node_id, model in self.energy_models.items():
            # Base idle power (minimal for modern efficient equipment)
            idle_power = 10.0  # 10W idle
            
            # Traffic-dependent power dominates (realistic for active routers/switches)
            # High-performance routers: 200-800W per Gbps of forwarding capacity
            traffic_load = node_traffic_loads.get(node_id, 0.0)
            traffic_power = traffic_load * 500.0  # 500W per Gbps
            
            # Total power = idle + traffic-dependent
            total_power_watts = idle_power + traffic_power
            
            # Convert to energy in kWh
            energy_kwh = (total_power_watts / 1000.0) * (duration / 3600.0)
            
            # Calculate carbon emission
            carbon_intensity = carbon_intensities.get(node_id, 350)
            carbon_emission = energy_kwh * carbon_intensity
            total_carbon += carbon_emission
        
        return total_carbon

    
    def snapshot(self):
        return {
            'timestamp': len(self.energy_history),
            'total_consumed': self.total_energy_consumed,
            'energy_ratios': self.get_all_energy_ratios(),
            'total_power': self.get_total_power()
        }
    
    def record_snapshot(self):
        self.energy_history.append(self.snapshot())


if __name__ == "__main__":
    print("Energy Model Test")
    print("=" * 50)
    
    manager = NetworkEnergyManager(5)
    manager.initialize_nodes(['grid', 'grid', 'battery', 'solar', 'grid'])
    
    print("\nInitial state:")
    for nid, ratio in manager.get_all_energy_ratios().items():
        print(f"  Node {nid}: {ratio*100:.1f}% energy remaining")
    
    for _ in range(10):
        manager.update_node_state(0, 'TX', 1.0)
        manager.update_node_state(1, 'RX', 1.0)
        manager.process_transmission(2, 3, 1500)
    
    print(f"\nAfter 10 transmissions:")
    print(f"  Total energy consumed: {manager.total_energy_consumed:.2f} J")
    print(f"  Total power: {manager.get_total_power():.2f} W")
    
    carbon_intensities = {i: 300 + i * 50 for i in range(5)}
    carbon = manager.calculate_carbon_emission(carbon_intensities, 10.0)
    print(f"  Carbon emission: {carbon:.4f} gCO2")
    
    print("\nEnergy model test passed.")
