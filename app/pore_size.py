import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path if needed
sys.path.insert(0, str(Path.cwd().parent / "src"))

# # Import mcp_lammps modules
from .data_handler import DataHandler
# from mcp_lammps.utils.openff_utils import openff_forcefield, OPENFF_AVAILABLE

# For data display
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Setup logging
logging.basicConfig(level=logging.WARNING)

print("✓ Imports successful!")
print(f"✓ Working directory: {Path.cwd()}")

print("=" * 80)
print("EXAMPLE 1: Pore size of r==5")
print("=" * 80)

# Define system parameters
ethanol_params = {
    "smiles": "CCO",
    "molecule_count": 100,
    "target_density": 0.789,  # g/cm³ at 298 K
    "name": "ethanol_pure"
}

print(f"\nSystem Parameters:")
print(f"  SMILES: {ethanol_params['smiles']}")
print(f"  Number of molecules: {ethanol_params['molecule_count']}")
print(f"  Target density: {ethanol_params['target_density']} g/cm³")

# Create liquid box
print(f"\nCreating liquid box...")
try:
    molecules = [{
        "smiles": ethanol_params['smiles'],
        "count": ethanol_params['molecule_count'],
        "name": "ethanol"
    }]
    
    data_file = data_handler.create_liquid_box_file(
        molecules=molecules,
        target_density=ethanol_params['target_density'],
        box_type="cubic"
    )
    
    print(f"✓ Liquid box created successfully!")
    print(f"  Data file: {data_file.name}")
    
    # Load metadata
    metadata_file = data_file.with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n  System Information:")
        if 'system_info' in metadata:
            sys_info = metadata['system_info']
            print(f"    Total atoms: {sys_info.get('n_atoms', 'N/A')}")
            print(f"    Total bonds: {sys_info.get('n_bonds', 'N/A')}")
            print(f"    Total angles: {sys_info.get('n_angles', 'N/A')}")
            print(f"    Total dihedrals: {sys_info.get('n_dihedrals', 'N/A')}")
        
        if 'charge_information' in metadata:
            charge_info = metadata['charge_information']
            total_charge = charge_info.get('total_system_charge', 0)
            print(f"    Total charge: {total_charge:.6f} e")
            if abs(total_charge) < 1e-4:
                print(f"    ✓ System is electrically neutral")
        
        # Store results
        results['example1'] = {
            'name': 'Pure Ethanol',
            'data_file': str(data_file),
            'metadata': metadata,
            'params': ethanol_params
        }
    
    print(f"\n✓ Example 1 completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

