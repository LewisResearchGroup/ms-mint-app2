"""
Python implementation of mass calculation routines.
Calculates exact mass from empirical formula with ionization mode support.
"""

import re
from typing import Dict, Tuple
from enum import Enum


class IonizationType(Enum):
    """Ionization type for mass spectrometry"""
    ESI = 0  # Electrospray Ionization (adds/removes protons)
    EI = 1   # Electron Impact (adds/removes electrons)


# Physical constants
ELECTRON_MASS = 0.00054857971
H_MASS = 1.0078250321
PROTON_MASS = H_MASS - ELECTRON_MASS  # 1.007276534


# Element masses
ELEMENT_MASSES = {
    'H': 1.007825,
    'He': 4.002603,
    'Li': 7.016005,
    'Be': 9.012183,
    'B': 11.009305,
    'C': 12.0,
    'N': 14.003074,  # N14_MASS from constants
    'O': 15.994915,
    'F': 18.998403,
    'Ne': 19.992439,
    'Na': 22.98977,
    'Mg': 23.985045,
    'Al': 26.981541,
    'Si': 27.976928,
    'P': 30.973763,
    'S': 31.972072,
    'Cl': 34.968853,
    'K': 38.963708,
    'Ca': 39.962591,
    'Ti': 47.947947,
    'V': 50.943963,
    'Cr': 51.940510,
    'Mn': 54.938046,
    'Fe': 55.934939,
    'Ni': 57.935347,
    'Co': 58.933198,
    'Cu': 62.929599,
    'Zn': 63.929145,
    'As': 74.921596,
    'Se': 79.916521,
    'Br': 78.918336,
    'Kr': 83.911506,
    'Rb': 84.9118,
    'Sr': 87.905625,
    'Y': 88.905856,
    'Mo': 97.905405,
    'Ru': 101.90434,
    'Rh': 102.905503,
    'Pd': 105.903475,
    'Ag': 106.905095,
    'Cd': 113.903361,
    'I': 126.904477,
    'Te': 129.906229,
    'Xe': 131.904148,
    'Ba': 137.905236,
    'La': 138.906355,
    'Pr': 140.907657,
    'Nd': 141.907731,
    'Sm': 151.919741,
    'Gd': 157.924111,
    'Tb': 158.92535,
    'Dy': 163.929183,
    'Lu': 174.940785,
    'Yb': 173.938873,
    'Hf': 179.946561,
    'Ta': 180.948014,
    'W': 183.950953,
    'Re': 186.955765,
    'Pt': 194.964785,
    'Au': 196.966560,
    'Tl': 204.97441,
    'Th': 232.038054,
    'Am': 241.056829,
}


class MassCalculator:
    """
    Calculate exact mass from empirical formula.
    """
    
    def __init__(self, ionization_type: IonizationType = IonizationType.ESI):
        """
        Initialize mass calculator.
        
        Args:
            ionization_type: ESI (default) or EI ionization
        """
        self.ionization_type = ionization_type
    
    @staticmethod
    def get_composition(formula: str) -> Dict[str, int]:
        """
        Parse empirical formula into element composition.
        Replicates MassCalculator::getComposition()
        
        Args:
            formula: Chemical formula string (e.g., "C6H12O6")
            
        Returns:
            Dictionary mapping element symbols to counts
            
        Example:
            >>> get_composition("C6H12O6")
            {'C': 6, 'H': 12, 'O': 6}
        """
        atoms = {}
        
        # Parse formula using regex to match element symbol + optional count
        # Pattern: Capital letter + optional lowercase + optional digits
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        for element, count in matches:
            if element:  # Skip empty matches
                count = int(count) if count else 1
                atoms[element] = atoms.get(element, 0) + count
        
        return atoms
    
    @staticmethod
    def get_element_mass(element: str) -> float:
        """
        Get monoisotopic mass of an element.
        Replicates MassCalculator::getElementMass()
        
        Args:
            element: Element symbol (e.g., "C", "H", "O")
            
        Returns:
            Monoisotopic mass in daltons (Da)
        """
        return ELEMENT_MASSES.get(element, 0.0)
    
    @staticmethod
    def compute_neutral_mass(formula: str) -> float:
        """
        Calculate neutral mass from empirical formula.
        Replicates MassCalculator::computeNeutralMass()
        
        Args:
            formula: Chemical formula string (e.g., "C6H12O6")
            
        Returns:
            Neutral mass in daltons (Da)
            
        Example:
            >>> compute_neutral_mass("C6H12O6")
            180.06339
        """
        atoms = MassCalculator.get_composition(formula)
        
        mass = 0.0
        for element, count in atoms.items():
            mass += MassCalculator.get_element_mass(element) * count
        
        return mass
    
    def adjust_mass(self, mass: float, charge: int) -> float:
        """
        Adjust mass for charge state based on ionization type.
        Replicates MassCalculator::adjustMass()
        
        Args:
            mass: Neutral mass in daltons
            charge: Charge state (positive, negative, or 0)
            
        Returns:
            Mass-to-charge ratio (m/z)
            
        Example:
            >>> calc = MassCalculator(IonizationType.ESI)
            >>> calc.adjust_mass(180.06339, 1)  # [M+H]+
            181.070666534
        """
        if mass == 0:
            return 0.0
        
        # EI mode: removes/adds electrons
        if self.ionization_type == IonizationType.EI and charge != 0:
            return (mass - charge * ELECTRON_MASS) / abs(charge)
        
        # Neutral (no charge)
        if charge == 0:
            return mass
        
        # ESI mode (default): adds/removes protons
        else:
            return (mass + charge * PROTON_MASS) / abs(charge)
    
    def compute_mass(self, formula: str, charge: int) -> float:
        """
        Calculate m/z from empirical formula and charge state.
        Replicates MassCalculator::computeMass()
        
        This is the main function that combines formula parsing,
        mass calculation, and charge adjustment.
        
        Args:
            formula: Chemical formula string (e.g., "C6H12O6")
            charge: Charge state (e.g., +1, -1, +2, 0)
            
        Returns:
            Mass-to-charge ratio (m/z)
            
        Examples:
            >>> calc = MassCalculator(IonizationType.ESI)
            >>> calc.compute_mass("C6H12O6", 1)   # [M+H]+ in positive mode
            181.070666534
            >>> calc.compute_mass("C6H12O6", -1)  # [M-H]- in negative mode
            179.055923466
            >>> calc.compute_mass("C6H12O6", 0)   # Neutral mass
            180.06339
        """
        neutral_mass = self.compute_neutral_mass(formula)
        return self.adjust_mass(neutral_mass, charge)
    
    @staticmethod
    def pretty_name(c: int, h: int, n: int, o: int, p: int, s: int) -> str:
        """
        Generate formatted formula string from element counts.
        Replicates MassCalculator::prettyName()
        
        Args:
            c: Carbon count
            h: Hydrogen count
            n: Nitrogen count
            o: Oxygen count
            p: Phosphorus count
            s: Sulfur count
            
        Returns:
            Formatted formula string (e.g., "C6H12O6")
        """
        name = ""
        
        if c != 0:
            name += "C"
            if c > 1:
                name += str(c)
        
        if h != 0:
            name += "H"
            if h > 1:
                name += str(h)
        
        if n != 0:
            name += "N"
            if n > 1:
                name += str(n)
        
        if o != 0:
            name += "O"
            if o > 1:
                name += str(o)
        
        if p != 0:
            name += "P"
            if p > 1:
                name += str(p)
        
        if s != 0:
            name += "S"
            if s > 1:
                name += str(s)
        
        return name


# Convenience functions for common use cases
def calculate_mz_esi_positive(formula: str, charge: int = 1) -> float:
    """
    Calculate m/z in ESI positive mode [M+H]+
    
    Args:
        formula: Chemical formula
        charge: Charge state (default: +1)
        
    Returns:
        m/z value
    """
    calc = MassCalculator(IonizationType.ESI)
    return calc.compute_mass(formula, charge)


def calculate_mz_esi_negative(formula: str, charge: int = -1) -> float:
    """
    Calculate m/z in ESI negative mode [M-H]-
    
    Args:
        formula: Chemical formula
        charge: Charge state (default: -1)
        
    Returns:
        m/z value
    """
    calc = MassCalculator(IonizationType.ESI)
    return calc.compute_mass(formula, charge)


def calculate_neutral_mass(formula: str) -> float:
    """
    Calculate neutral mass from formula.
    
    Args:
        formula: Chemical formula
        
    Returns:
        Neutral mass in Da
    """
    return MassCalculator.compute_neutral_mass(formula)


# Example usage and testing
if __name__ == "__main__":
    
    # Example compounds from KNOWNS.csv
    test_compounds = [
        ("C6H12O6", "Glucose", 1),
        ("C10H16N5O13P3", "ATP", -1),
        ("C21H36N7O16P3S", "Coenzyme A", -1),
        ("C23H38N7O17P3S", "Acetyl-CoA", -1),
        ("C9H15N2O14P3", "UTP", -1),
    ]
    
    calc_esi = MassCalculator(IonizationType.ESI)
    
    for formula, name, charge in test_compounds:
        neutral = calc_esi.compute_neutral_mass(formula)
        mz = calc_esi.compute_mass(formula, charge)
        mode = f"[M+H]+" if charge > 0 else f"[M-H]-"
        
        print(f"\n{name}")
        print(f"  Formula: {formula}")
        print(f"  Neutral mass: {neutral:.4f} Da")
        print(f"  m/z ({mode}): {mz:.4f}")
    
    print("\n" + "=" * 60)
    print("\nFormula parsing examples:")
    examples = ["C6H12O6", "C10H16N5O13P3", "NaCl", "CaCO3"]
    for formula in examples:
        composition = MassCalculator.get_composition(formula)
        print(f"{formula:20s} -> {composition}")
    
    print("\n" + "=" * 60)
    print(f"  ATP [M-H]- calculated: {calc_esi.compute_mass('C10H16N5O13P3', -1):.6f}")
    print(f"  Expected from test: ~505.988")
