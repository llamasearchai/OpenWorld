from .device import SolarCellDevice
from ...utils.units import ureg, u
from scipy.constants import e, k
from scipy.integrate import trapz

class SolarSimulation:
    def __init__(self, device: Union[SolarCellDevice, str] = "standard_silicon"):
        self.sim_id = str(uuid.uuid4())
        self.device = device if isinstance(device, SolarCellDevice) else \
                     SolarCellDevice.create_standard_device(device)
        self.results = None
        self._init_physical_constants()
        
    def _init_physical_constants(self):
        self.q = e  # Electron charge
        self.kT = k * 300  # Thermal energy at 300K
        self.J0 = 1e-12  # Reverse saturation current density
        self.Jsc = 0.034  # Short-circuit current density (A/cm²)
        
    def run_iv_curve(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        voltages = np.linspace(
            conditions['voltage_min_V'],
            conditions['voltage_max_V'],
            conditions['voltage_steps']
        )
        
        # Calculate current using the two-diode model
        current = self._calculate_current_density(voltages, conditions) * self.device.active_area.to('cm^2').magnitude
        
        # Calculate power and find maximum power point
        power = voltages * current
        max_power_idx = np.argmax(power)
        
        self.results = {
            'voltage_V': voltages.tolist(),
            'current_A': current.tolist(),
            'power_W': power.tolist(),
            'efficiency': self._calculate_efficiency(power[max_power_idx], conditions),
            'fill_factor': self._calculate_fill_factor(voltages, current),
            'max_power_point': {
                'voltage': float(voltages[max_power_idx]),
                'current': float(current[max_power_idx]),
                'power': float(power[max_power_idx])
            }
        }
        return self.results
        
    def _calculate_current_density(self, voltage: np.ndarray, conditions: Dict[str, Any]) -> np.ndarray:
        # Two-diode model implementation
        Vt = self.kT / self.q  # Thermal voltage
        Jph = self.Jsc * (conditions['light_intensity_W_m2'] / 1000)
        
        # Diode 1 (ideal) and Diode 2 (recombination)
        J1 = self.J0 * (np.exp(voltage / Vt) - 1)
        J2 = (self.J0/2) * (np.exp(voltage / (2*Vt)) - 1)
        
        # Series resistance effect
        Rs = 0.1  # Series resistance (ohms)
        J = Jph - J1 - J2 - (voltage / Rs)
        
        return np.where(J > 0, J, 0)  # Clip negative current
        
    def _calculate_efficiency(self, max_power: float, conditions: Dict[str, Any]) -> float:
        input_power = conditions['light_intensity_W_m2'] * (self.device.active_area.to('m^2').magnitude)
        return (max_power / input_power) * 100
        
    def _calculate_fill_factor(self, voltage: np.ndarray, current: np.ndarray) -> float:
        voc_idx = np.argmin(np.abs(current))
        isc_idx = np.argmin(np.abs(voltage))
        voc = voltage[voc_idx]
        isc = current[isc_idx]
        pmax = np.max(voltage * current)
        return pmax / (voc * isc) 