"""
Solar cell device model for OpenWorld, defining the layer stack and properties.
"""

from typing import Dict, List, Any as TypingAny, Optional, Tuple, Union # Renamed Any to TypingAny
import numpy as np
from dataclasses import dataclass, field
from pint import Quantity # For unit handling

from ...utils.logging import get_logger
from ...utils.units import ureg, u, convert_to_base_units, ensure_quantity # Assuming ureg and u are defined here
from ...utils.exceptions import SolarError, ConfigurationError
from .material import MaterialProperties # Assuming MaterialProperties is in the same directory

logger = get_logger(__name__)

@dataclass
class Layer:
    """Represents a single layer in a solar cell device stack."""
    name: str
    material: MaterialProperties
    thickness: Quantity # Thickness of the layer (e.g., in nm or um)
    layer_type: str = "absorber"  # E.g., absorber, window, BSF, ETL, HTL, contact, ARC
    metadata: Dict[str, TypingAny] = field(default_factory=dict) # For additional layer-specific info

    def __post_init__(self):
        if not isinstance(self.material, MaterialProperties):
            raise ConfigurationError(f"Layer '{self.name}' must be initialized with a MaterialProperties instance, got {type(self.material)}.")
        self.thickness = ensure_quantity(self.thickness, u.nm) # Default to nm if no unit provided
        if self.thickness.magnitude <= 0:
            raise ConfigurationError(f"Layer '{self.name}' thickness must be positive, got {self.thickness.to_compact()}.")
        logger.debug(f"Layer '{self.name}' created: Material='{self.material.name}', Thickness={self.thickness.to_compact()}, Type='{self.layer_type}'.")

    def __repr__(self) -> str:
        return f"Layer(name='{self.name}', material='{self.material.name}', thickness={self.thickness.to_compact()}, type='{self.layer_type}')"

class SolarCellDevice:
    """
    Multi-layer solar cell device model.
    
    This class represents the physical structure of a solar cell, comprising multiple layers,
    each with defined material properties and thickness. It also manages contacts and overall device properties.
    """
    
    def __init__(self, name: str = "SolarCellDevice", description: Optional[str] = None):
        """
        Initialize a solar cell device.
        
        Args:
            name: Descriptive name for the device.
            description: Optional longer description of the device.
        """
        self.name = name
        self.description = description or f"A multi-layer solar cell device named '{name}'."
        self.layers: List[Layer] = []
        self.contacts: Dict[str, Dict[str, TypingAny]] = { # Properties for front and back contacts
            "front": {"work_function": 4.5 * u.eV, "type": "ideal_ohmic"},
            "back": {"work_function": 4.2 * u.eV, "type": "ideal_ohmic"}
        }
        self.active_area: Quantity = 1.0 * u.cm**2 # Default active area
        self.operating_temperature: Quantity = 300 * u.K # Default operating temperature
        
        logger.info(f"Initialized SolarCellDevice: '{self.name}'")
    
    def add_layer(self, 
                  name: str, 
                  material: Union[MaterialProperties, str], 
                  thickness: Union[float, Quantity], 
                  layer_type: str = "absorber",
                  position: Optional[int] = None, # Insert at position, None or -1 appends
                  doping_type: Optional[str] = None, # \'n\' or \'p\' if material is str
                  doping_concentration_cm3: Optional[float] = None, # if material is str
                  metadata: Optional[Dict[str, TypingAny]] = None) -> None:
        """
        Add a layer to the device structure.
        
        Args:
            name: Unique name for the layer.
            material: MaterialProperties instance or string name to load from MaterialProperties.from_library().
            thickness: Layer thickness (float assumes nm, or provide Quantity with units).
            layer_type: Functional type of the layer (e.g., absorber, window, contact, ETL, HTL).
            position: Index to insert the layer at. If None or out of bounds, appends to the end.
            doping_type: If material is a string, specifies doping type (\'n\' or \'p\').
            doping_concentration_cm3: If material is a string, specifies doping concentration (cm^-3).
            metadata: Additional key-value data for the layer.
        """
        if any(layer.name == name for layer in self.layers):
            raise ConfigurationError(f"Layer with name '{name}' already exists in device '{self.name}'.")

        if isinstance(material, str):
            try:
                material_instance = MaterialProperties.from_library(material, doping_type, doping_concentration_cm3)
            except Exception as e:
                raise ConfigurationError(f"Failed to load material '{material}' from library: {e}")
        elif isinstance(material, MaterialProperties):
            material_instance = material
        else:
            raise ConfigurationError(f"Invalid material type for layer '{name}'. Must be MaterialProperties or string name.")
        
        thickness_q = ensure_quantity(thickness, u.nm)
        
        layer = Layer(name, material_instance, thickness_q, layer_type, metadata or {})
        
        if position is None or not (0 <= position <= len(self.layers)):
            self.layers.append(layer)
            logger.debug(f"Appended layer '{name}' to device '{self.name}'. Total layers: {len(self.layers)}.")
        else:
            self.layers.insert(position, layer)
            logger.debug(f"Inserted layer '{name}' at position {position} in device '{self.name}'. Total layers: {len(self.layers)}.")

    def remove_layer(self, name: str) -> bool:
        """
        Remove a layer by its name.
        Args:
            name: Name of the layer to remove.
        Returns:
            True if layer was found and removed, False otherwise.
        """
        for i, layer in enumerate(self.layers):
            if layer.name == name:
                removed_layer = self.layers.pop(i)
                logger.info(f"Removed layer '{removed_layer.name}' from device '{self.name}'.")
                return True
        logger.warning(f"Layer '{name}' not found for removal in device '{self.name}'.")
        return False

    def get_layer(self, name: str) -> Optional[Layer]:
        """
        Get a layer by its name.
        Args:
            name: Name of the layer.
        Returns:
            The Layer object if found, else None.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def set_layer_thickness(self, name: str, new_thickness: Union[float, Quantity]) -> None:
        """
        Set the thickness of an existing layer.
        Args:
            name: Name of the layer to modify.
            new_thickness: New thickness (float assumes nm, or Quantity with units).
        """
        layer = self.get_layer(name)
        if layer:
            new_thickness_q = ensure_quantity(new_thickness, u.nm)
            if new_thickness_q.magnitude <= 0:
                raise ConfigurationError(f"New thickness for layer '{name}' must be positive, got {new_thickness_q.to_compact()}.")
            old_thickness = layer.thickness
            layer.thickness = new_thickness_q
            logger.info(f"Updated thickness of layer '{name}' from {old_thickness.to_compact()} to {layer.thickness.to_compact()}.")
        else:
            raise ConfigurationError(f"Layer '{name}' not found in device '{self.name}' to set thickness.")

    def set_contact_property(self, contact_side: str, property_name: str, value: TypingAny) -> None:
        """
        Set a property for a contact (e.g., work function, type).
        Args:
            contact_side: \'front\' or \'back\'.
            property_name: Name of the property to set (e.g., \'work_function\', \'type\').
            value: The value for the property (e.g., 4.2 * u.eV, \"schottky\").
        """
        if contact_side not in self.contacts:
            raise ConfigurationError(f"Invalid contact side '{contact_side}\'. Must be \'front\' or \'back\'.")
        if isinstance(value, (float, int)) and property_name == "work_function": 
            value_q = value * u.eV
        else:
            value_q = value 
        self.contacts[contact_side][property_name] = value_q
        logger.info(f"Set {contact_side} contact property '{property_name}' to {value_q}.")

    @property
    def total_thickness(self) -> Quantity:
        """
        Calculate the total thickness of all layers in the device.
        Returns:
            Total thickness as a Quantity (typically in nm or um).
        """
        if not self.layers:
            return 0 * u.nm
        total_th = sum(layer.thickness.to(u.nm) for layer in self.layers)
        return total_th.to_compact()

    def get_layer_positions(self, reference_unit: str = 'nm') -> Dict[str, Tuple[Quantity, Quantity]]:
        """
        Get the start and end position (depth) of each layer from the front surface (z=0).
        Args:
            reference_unit: Unit for position values (e.g., \'nm\', \'um\').
        Returns:
            Dictionary mapping layer names to (start_depth, end_depth) Quantities.
        """
        positions: Dict[str, Tuple[Quantity, Quantity]] = {}
        current_depth = 0.0 * ureg(reference_unit)
        
        for layer in self.layers:
            start_depth = current_depth
            end_depth = start_depth + layer.thickness.to(reference_unit)
            positions[layer.name] = (start_depth, end_depth)
            current_depth = end_depth
            
        return positions

    def get_absorber_layers(self) -> List[Layer]:
        """
        Get all layers designated as \'absorber\' type.
        Returns:
            List of absorber Layer objects.
        """
        return [layer for layer in self.layers if layer.layer_type.lower() == "absorber"]

    def get_material_at_depth(self, depth: Quantity) -> Optional[MaterialProperties]:
        """
        Get the material at a specific depth from the front surface.
        Args:
            depth: Depth from the front surface (Quantity with length unit).
        Returns:
            MaterialProperties of the layer at that depth, or None if depth is outside device.
        """
        depth_nm = depth.to(u.nm).magnitude
        current_pos_nm = 0.0
        for layer in self.layers:
            layer_thickness_nm = layer.thickness.to(u.nm).magnitude
            if current_pos_nm <= depth_nm < current_pos_nm + layer_thickness_nm:
                return layer.material
            current_pos_nm += layer_thickness_nm
        return None

    def to_dict(self) -> Dict[str, TypingAny]:
        """
        Convert the device structure to a serializable dictionary.
        Returns:
            A dictionary representation of the SolarCellDevice.
        """
        device_dict = {
            "name": self.name,
            "description": self.description,
            "active_area": {"magnitude": self.active_area.magnitude, "unit": str(self.active_area.units)},
            "operating_temperature": {"magnitude": self.operating_temperature.magnitude, "unit": str(self.operating_temperature.units)},
            "total_thickness": {"magnitude": self.total_thickness.magnitude, "unit": str(self.total_thickness.units)},
            "layers": [],
            "contacts": {cs: {k: (v.to_dict() if hasattr(v, 'to_dict') else ({"magnitude": v.magnitude, "unit": str(v.units)} if isinstance(v, Quantity) else str(v))) for k,v in props.items()} for cs, props in self.contacts.items()},
        }
        for layer in self.layers:
            device_dict["layers"].append({
                "name": layer.name,
                "material_name": layer.material.name,
                "thickness": {"magnitude": layer.thickness.magnitude, "unit": str(layer.thickness.units)},
                "layer_type": layer.layer_type,
                "metadata": layer.metadata
            })
        return device_dict

    @classmethod
    def from_dict(cls, data: Dict[str, TypingAny]) -> 'SolarCellDevice':
        """
        Create a SolarCellDevice instance from a dictionary representation.
        Args:
            data: Dictionary containing device data.
        Returns:
            A SolarCellDevice instance.
        """
        device = cls(name=data.get("name", "SolarCellDevice_from_dict"), 
                     description=data.get("description"))
        
        if "active_area" in data and isinstance(data["active_area"], dict):
            device.active_area = data["active_area"]["magnitude"] * ureg(data["active_area"]["unit"])
        if "operating_temperature" in data and isinstance(data["operating_temperature"], dict):
            device.operating_temperature = data["operating_temperature"]["magnitude"] * ureg(data["operating_temperature"]["unit"])
        
        for side, props in data.get("contacts", {}).items():
            for prop_name, prop_val_data in props.items():
                val = prop_val_data
                if isinstance(prop_val_data, dict) and "magnitude" in prop_val_data and "unit" in prop_val_data: 
                    val = prop_val_data["magnitude"] * ureg(prop_val_data["unit"])
                elif isinstance(prop_val_data, str) and "eV" in prop_val_data and prop_name == "work_function": 
                    try: val = float(prop_val_data.replace("eV","").strip()) * u.eV
                    except: pass 
                device.contacts[side][prop_name] = val

        for layer_data in data.get("layers", []):
            material_name = layer_data.get("material_name")
            if not material_name:
                raise ConfigurationError(f"Layer data missing 'material_name': {layer_data}")
            
            try:
                material_instance = MaterialProperties.from_library(material_name)
            except Exception as e:
                logger.error(f"Could not load material '{material_name}' from library for layer '{layer_data.get('name')}': {e}. Skipping layer.")
                continue

            thickness_data = layer_data.get("thickness")
            if not isinstance(thickness_data, dict) or "magnitude" not in thickness_data or "unit" not in thickness_data:
                raise ConfigurationError(f"Layer data missing valid thickness: {layer_data}")
            thickness_q = thickness_data["magnitude"] * ureg(thickness_data["unit"])

            device.add_layer(
                name=layer_data.get("name", f"Layer_{len(device.layers)}"),
                material=material_instance,
                thickness=thickness_q,
                layer_type=layer_data.get("layer_type", "unknown"),
                metadata=layer_data.get("metadata", {})
            )
        return device

    @classmethod
    def create_standard_device(cls, device_type: str = "SiliconPNJunction", 
                             absorber_thickness_nm: float = 300.0) -> 'SolarCellDevice':
        """
        Create a standard, predefined solar cell device structure.
        
        Args:
            device_type: Type of device to create (e.g., \"SiliconPNJunction\", \"PerovskitePIN\").
            absorber_thickness_nm: Thickness of the main absorber layer in nanometers.
            
        Returns:
            A configured SolarCellDevice instance.
        """
        device_name = f"Standard_{device_type}_Cell"
        device = cls(name=device_name)
        abs_thick = absorber_thickness_nm * u.nm

        if device_type == "SiliconPNJunction":
            device.add_layer(name="ARC", material="ITO", thickness=80*u.nm, layer_type="antireflection_coating")
            device.add_layer(name="Emitter", material=MaterialProperties.from_library("Silicon", doping_type='n', doping_concentration_cm3=1e19), 
                             thickness=0.2*u.um, layer_type="emitter")
            device.add_layer(name="BaseAbsorber", material=MaterialProperties.from_library("Silicon", doping_type='p', doping_concentration_cm3=1e16), 
                             thickness=abs_thick, layer_type="absorber")
            device.add_layer(name="BSF", material=MaterialProperties.from_library("Silicon", doping_type='p', doping_concentration_cm3=1e18), 
                             thickness=0.5*u.um, layer_type="back_surface_field")

        elif device_type == "PerovskitePIN":
            device.add_layer(name="FTO_Substrate", material="ITO", thickness=200*u.nm, layer_type="transparent_conductor")
            
            etl_mat = MaterialProperties.from_library("ITO") # Using ITO as a base for a generic ETL
            etl_mat.name = "Generic_ETL"
            etl_mat.bandgap = 3.2 * u.eV 
            etl_mat.electron_affinity = 4.0 * u.eV 
            # Make it n-type if not already highly n-doped like ITO
            if etl_mat.donor_concentration.magnitude < 1e18:
                 etl_mat.donor_concentration = 1e19 / u.cm**3
                 etl_mat.acceptor_concentration = 0 / u.cm**3
            device.add_layer(name="ETL", material=etl_mat, thickness=30*u.nm, layer_type="electron_transport_layer")
            
            device.add_layer(name="PerovskiteAbsorber", material=MaterialProperties.from_library("Perovskite_MAPbI3"), 
                             thickness=abs_thick, layer_type="absorber")
            
            htl_mat = MaterialProperties.from_library("Perovskite_MAPbI3") # Base for modification, then make p-type
            htl_mat.name = "Generic_HTL"
            htl_mat.acceptor_concentration = 5e18 / u.cm**3 
            htl_mat.donor_concentration = 0 / u.cm**3
            htl_mat.bandgap = 2.9 * u.eV 
            htl_mat.electron_affinity = 3.5 * u.eV # Adjust affinity for p-type alignment
            device.add_layer(name="HTL", material=htl_mat, thickness=50*u.nm, layer_type="hole_transport_layer")

        else:
            raise ConfigurationError(f"Unknown standard device type: '{device_type}'. Implemented: SiliconPNJunction, PerovskitePIN.")
            
        logger.info(f"Created standard device: '{device.name}\' with total thickness {device.total_thickness.to_compact()}.")
        return device

    def __repr__(self) -> str:
        return f"SolarCellDevice(name='{self.name}\', layers={len(self.layers)}, total_thickness={self.total_thickness.to_compact()})" 