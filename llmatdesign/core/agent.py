import os
import re
import copy
import random
import numpy as np
from time import time

from pathlib import Path
from mp_api.client import MPRester

import ase
import ase.io
from ase import Atoms
from ase.data import atomic_numbers
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.composition import Composition

from matdeeplearn.common.ase_utils import MDLCalculator

from llmatdesign.modules.structure_optimization import StructureOptimizer
VALID_SYMBOLS = set(atomic_numbers.keys()) #global variable for valid symbol sanity check
class Agent:
    def __init__(
        self, 
        llm, 
        save_path=None,
        forcefield_config_path=None,
        bandgap_config_path=None,
        formation_energy_config_path=None,
        mp_api_key=None,
    ):
        self.llm = llm
        self.save_path = Path("./outputs/") if save_path is None else Path(save_path)
        self.forcefield_config_path = forcefield_config_path
        self.bandgap_config_path = bandgap_config_path
        self.formation_energy_config_path = formation_energy_config_path

        self.is_success = False
        self.mp_api_key = os.environ.get("HykOG4IhaN8Xi2kH3dq0lr42nLpcMBZE") if mp_api_key is None else mp_api_key

        # set up the force field calculator
        if self.forcefield_config_path is not None:
            self.calculator = MDLCalculator(self.forcefield_config_path)
            self.structure_optimizer = StructureOptimizer(self.calculator)
        else:
            self.calculator = None
            self.structure_optimizer = None
        
        # set up the band gap calculator
        if self.bandgap_config_path is not None:
            self.bandgap_calculator = MDLCalculator(self.bandgap_config_path)
        else:
            self.bandgap_calculator = None

        # set up the formation energy calculator
        if self.formation_energy_config_path is not None:
            self.formation_energy_calculator = MDLCalculator(self.formation_energy_config_path)
        else:
            self.formation_energy_calculator = None
    
    # report
    def report(self):
        pass
    
    def _ask(self, prompt_template, context, question):
        prompt = prompt_template.replace('<context>', context).replace('<question>', question)
        answer = self.llm.ask(prompt)

        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return answer

    #### TOOLS FOR THE AGENT ####

    # query the materials project API to complete the task
    def query_materials_project(self, composition, target_property):
        supported_properties = [
            "material_id", "structure", "formation_energy_per_atom",
            "band_gap", "energy_above_hull", "symmetry", "elements", "composition"
        ]

        # check if the target property is supported
        try:
            assert target_property in supported_properties
        except:
            raise ValueError(f"Invalid target property: {target_property}")

        # check if the composition is valid
        try:
            _composition = Composition(composition)
        except:
            raise ValueError(f"Invalid composition: {composition}")
        
        # query the materials project API
        with MPRester(self.mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=composition,
                fields=supported_properties
            )

        if len(docs) == 0:
            return [False, None]
        elif len(docs) == 1:
            # return ase.Atoms object if the target property is "structure"
            if target_property == "structure":
                pmg_structure = docs[0].structure
                #print(pmg_structure) #DEBUGGING STATEMENT
                ase_atoms = AseAtomsAdaptor.get_atoms(pmg_structure, msonable=False)
                #print(ase_atoms)
                return [True, ase_atoms]
            return [True, getattr(docs[0], target_property)]
        else:
            # multiple documents found, return the one with the lowest formation energy per atom
            l = [getattr(doc, "formation_energy_per_atom") for doc in docs]
            idx = np.argmin(l)

            # return ase.Atoms object if the target property is "structure"
            if target_property == "structure":
                pmg_structure = docs[idx].structure
                #print(pmg_structure) #DEBUGGING STATEMENT
                ase_atoms = AseAtomsAdaptor.get_atoms(pmg_structure, msonable=False)
                #print(ase_atoms)
                return [True, ase_atoms]
            return [True, getattr(docs[idx], target_property)]

    def optimize_and_calculate(self, atoms, calculation_type="formation_energy"):
        try:
            assert self.calculator is not None
        except:
            raise ValueError("Calculator not set up")
        
        device = 'cpu'
        initial_atoms = [atoms]
        optimized_atoms = []

        if len(atoms) == 1:
            # no need to optimize single-atom structure
            optimized_atoms = [atoms]
        else:
            tic = time()
            times = []

            for idx, atoms in enumerate(initial_atoms):
                atoms.set_calculator(self.calculator)
                to_optim = copy.deepcopy(atoms)

                optimized, time_per_step = self.structure_optimizer.optimize(to_optim)

                times.append(time_per_step)
                optimized_atoms.append(optimized)
        
            toc = time()
            print(f"Optimized {len(initial_atoms)} structures in {toc - tic:.2f} s")
        real_structure= False
        uncertainty=0.0
        optimized_structure= optimized_atoms[0]
        new_chemical_formula = optimized_structure.get_chemical_formula()
        _, structure_from_mp = self.query_materials_project(new_chemical_formula, 'structure')
        if structure_from_mp is not None:
            real_structure=True
            _, band_gap = self.query_materials_project(structure_from_mp.get_chemical_formula(), 'band_gap')
            print("MP-API CALL SUCCESSFUL")
            band_gap_uncertainty=0.0
            return optimized_structure, band_gap, band_gap_uncertainty, real_structure
        else:
            

                print("OPENING A ROAD TO NEW DISCOVERY :)........")
                if calculation_type == "formation_energy" and self.formation_energy_calculator is not None:
                    #v= self.formation_energy_calculator.calculate(optimized_atoms[0])
                    val, val_uncertainty =self.formation_energy_calculator.direct_calculate_energy(optimized_atoms[0])
                    return optimized_atoms[0], val, val_uncertainty, real_structure
                elif calculation_type == "band_gap" and self.bandgap_calculator is not None:
                    #v= self.bandgap_calculator.calculate(optimized_atoms[0])
                    val, val_uncertainty = self.bandgap_calculator.direct_calculate_band_gap(optimized_atoms[0])
                    return optimized_atoms[0], val, val_uncertainty, real_structure
                else:
                    raise NotImplementedError
    
    def is_within_threshold(self, calculated_value, target_value, threshold=10):
        return abs(calculated_value - target_value) / abs(target_value) * 100 <= threshold

    def perform_modification(self, structure, modification, calculation_type):
        if isinstance(modification, str):
            from ast import literal_eval
            modification, reason = literal_eval(modification)

        modification_type = modification[0]

        chemical_symbols = structure.get_chemical_symbols()
        cell = structure.get_cell()
        positions = structure.get_positions()
        # helper to validate and build Atoms
        def build_atoms(symbols, pos, cell):
            # 1) validate each symbol is real
            for s in symbols:
                if s not in VALID_SYMBOLS:
                    raise ValueError(f"Invalid element from LLM: '{s}'")
            # 2) attempt to build, catching any ASE KeyError
            try:
                return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
            except KeyError as e:
                raise ValueError(f"ASE failed on symbol: {e}")
        
        if modification_type == "substitute":
            _, old_atom, new_atom = modification
            new_symbols = [new_atom if x == old_atom else x for x in chemical_symbols]
            
            new_structure = build_atoms(new_symbols, positions, cell)

        elif modification_type == "exchange":
            _, atom1, atom2 = modification
            new_symbols = [atom2 if x == atom1 else atom1 if x == atom2 else x for x in chemical_symbols]

            new_structure = build_atoms(new_symbols, positions, cell)

        elif modification_type == "add":
            _, atom = modification
            new_symbols = chemical_symbols + [atom]
            new_positions = np.vstack(
                (
                    positions,
                    self.random_3d_point_within_cell(cell[0], cell[1], cell[2])
                )
            )

            new_structure = build_atoms(new_symbols, new_positions, cell)

        elif modification_type == "remove":
            _, atom = modification
            new_symbols = [x for x in chemical_symbols if x != atom]
            new_positions = positions[[x != atom for x in chemical_symbols]]

            new_structure = build_atoms(new_symbols, new_positions, cell)

        else:
            raise ValueError(f"Invalid modification type: {modification_type}")
        # uncertainty=0.0
        # new_chemical_formula = new_structure.get_chemical_formula()
        # _, structure_from_mp = self.query_materials_project(new_chemical_formula, 'structure')
        # if structure_from_mp is not None:
        #   _, band_gap = self.query_materials_project(structure_from_mp.get_chemical_formula(), 'band_gap')
        #   uncertainty=0.0
        # else:

        return self.optimize_and_calculate(new_structure, calculation_type=calculation_type)

    @staticmethod
    def random_3d_point_within_cell(v1, v2, v3):
        # Compute the volume of the parallelepiped
        volume = np.abs(np.dot(v1, np.cross(v2, v3)))
        
        # Generate random numbers
        r1, r2 = np.random.rand(2)
        r3 = np.random.rand() * (1 - r1 - r2)
        
        # Calculate the random point
        random_point = r1 * v1 + r2 * v2 + r3 * v3
        
        return random_point