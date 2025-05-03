from typing import List
import logging

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from matdeeplearn.preprocessor.helpers import generate_node_features
import math
logging.basicConfig(level=logging.INFO)


class MDLCalculator(Calculator):
    """
    A neural networked based Calculator that calculates the energy, forces and stress of a crystal structure.
    """
    implemented_properties = ["energy", "forces", "stress","band_gap", "dielectric"]

    def __init__(self, config, rank='cpu'):
        """
        Initialize the MDLCalculator instance.

        Args:
        - config (str or dict): Configuration settings for the MDLCalculator.
        - rank (str): Rank of device the calculator calculates properties. Defaults to 'cuda:0'

        Raises:
        - AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        """
        Calculator.__init__(self)
        
        if isinstance(config, str):
            logging.info(f'MDLCalculator instantiated from config: {config}')
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
        elif isinstance(config, dict):
            logging.info('MDLCalculator instantiated from a dictionary.')
        else:
            raise NotImplementedError('Unsupported config type.')
        # Add MC dropout parameters
        self.mc_dropout = config["model"].get("mc_dropout", False)
        self.num_mc_samples = config["model"].get("num_mc_samples", 30)
        self.dropout_rate = config["model"].get("dropout_rate", 0.5)

        gradient = config["model"].get("gradient", False)
        otf_edge_index = config["model"].get("otf_edge_index", False)
        otf_edge_attr = config["model"].get("otf_edge_attr", False)
        self.otf_node_attr = config["model"].get("otf_node_attr", False)
        assert otf_edge_index and otf_edge_attr and gradient, "To use this calculator to calculate forces and stress, you should set otf_edge_index, oft_edge_attr and gradient to True."
        
        self.device = rank if torch.cuda.is_available() else 'cpu'
        self.models = MDLCalculator._load_model(config,'cpu')
        self.model=self.models[0]
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)
        #self.model.train()
    def calculate(self, atoms: Atoms, properties=implemented_properties, system_changes=None) -> None:
        """
        Calculate energy, forces, and stress for a given ase.Atoms object.

        Args:
        - atoms (ase.Atoms): The atomic structure for which calculations are to be performed.
        - properties (list): List of properties to calculate. Defaults to ['energy', 'forces', 'stress'].
        - system_changes: Not supported in the current implementation.
       
        Returns:
        - None: The results are stored in the instance variable 'self.results'.

        Note:
        - This method performs energy, forces, and stress calculations using a neural network-based calculator.
            The results are stored in the instance variable 'self.results' as 'energy', 'forces', and 'stress'.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        #self.model.train()  # Ensure dropout is active
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())

        data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
            z=atomic_numbers, structure_id=atoms.info.get('structure_id', None))
        
        # Generate node features
        if not self.otf_node_attr:
            generate_node_features(data, self.n_neighbors, device=self.device)
            data.x = data.x.to(torch.float32)
        
        data_list = [data]
        loader = DataLoader(data_list, batch_size=1)
        loader_iter = iter(loader)
        batch = next(loader_iter).to(self.device)

        assert not torch.isnan(batch.pos).any(), "NaNs in positions!"
        assert not torch.isinf(batch.pos).any(), "Infs in positions!"
        if self.mc_dropout:
            with torch.no_grad():
                uncertainty_results = self.model.mc_dropout_forward(batch, n_samples=self.num_mc_samples)
        #self.model.train()
        #out_list = []  
        # with torch.no_grad():

        #     for _ in range(self.num_mc_samples): #Monte Carlo dropout
        #         out = self.model(batch)
        #         #print(f"Output shape: {out[0]['output'].shape}")
        #         out_list.append({
        #         "output": out["output"].detach(),
        #         "pos_grad": out["pos_grad"].detach(),
        #         "cell_grad": out["cell_grad"].detach()
        #     })
    
        # energy = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)
        # energy_mean = energy.mean(dim=0).cpu().numpy()
        # energy_std = energy.std(dim=0).cpu().numpy() +1e-8
        # if not math.isnan(energy_std):
        #     print("Energy std: ", energy_std)
        # else:
        #     print("Energy std: NaN")
        # band_gap = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)
        # band_gap_mean = band_gap.mean(dim=0).cpu().numpy()
        # band_gap_std = band_gap.std(dim=0).cpu().numpy() +1e-8
        # if not math.isnan(band_gap_std):
        #     print("band_gap std: ", band_gap_std)
        # else:
        #     print("band_gap std: NaN")

        #dielectric = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)

        # forces = torch.stack([entry["pos_grad"] for entry in out_list]).mean(dim=0)
        # forces_mean = forces.mean(dim=0).cpu().numpy().reshape(-1, 3)
        # forces_std = forces.std(dim=0).cpu().numpy().reshape(-1, 3) 

        # stresses = torch.stack([entry["cell_grad"] for entry in out_list]).mean(dim=0)
        # stress_mean = stresses.mean(dim=0).cpu().numpy().reshape(-1, 3)
        # stress_std = stresses.std(dim=0).cpu().numpy().reshape(-1, 3)

        # Store results
        if self.mc_dropout:
            with torch.no_grad():
                uncertainty_results = self.model.mc_dropout_forward(batch, n_samples=self.num_mc_samples)
        
            # Extract results
            energy = uncertainty_results["output"]
            energy_mean = energy.detach().cpu().numpy()
            energy_std = uncertainty_results["uncertainty"].cpu().numpy() + 1e-8
            
            forces = uncertainty_results["pos_grad"]
            forces_mean = forces.detach().cpu().numpy().reshape(-1, 3)
            forces_std = uncertainty_results["pos_grad_uncertainty"].cpu().numpy().reshape(-1, 3)
            
            stresses = uncertainty_results["cell_grad"]
            stress_mean = stresses.detach().cpu().numpy().reshape(-1, 3)
            stress_std = uncertainty_results["cell_grad_uncertainty"].cpu().numpy().reshape(-1, 3)
        else:
            # Original forward pass
            self.model.eval()
            with torch.no_grad():
                out = self.model(batch)
            
            energy_mean = out["output"].detach().cpu().numpy()
            energy_std = torch.zeros_like(out["output"]).detach().cpu().numpy()
            forces_mean = out["pos_grad"].detach().cpu().numpy().reshape(-1, 3) if out["pos_grad"] is not None else None
            forces_std = np.zeros_like(forces_mean)
            stress_mean = out["cell_grad"].detach().cpu().numpy().reshape(-1, 3) if out["cell_grad"] is not None else None
            stress_std = np.zeros_like(stress_mean)

    # Store results with uncertainties
        self.results['energy'] = energy_mean.squeeze()
        self.results['energy_uncertainty'] = energy_std.squeeze()
        self.results['band_gap'] = energy_mean.squeeze()
        self.results['band_gap_uncertainty'] = energy_std.squeeze()
        self.results['forces'] = forces_mean
        self.results['forces_uncertainty'] = forces_std
        self.results['stress'] = stress_mean
        self.results['stress_uncertainty'] = stress_std
    def direct_calculate_energy(self,atoms:Atoms):
        self.calculate(atoms)
        return self.results['energy'], self.results['energy_uncertainty']
    
    def direct_calculate_band_gap(self,atoms:Atoms):
        self.calculate(atoms)
        return self.results['band_gap'], self.results['band_gap_uncertainty']
    
    # def direct_calculate_dielectric(self,atoms:Atoms):
    #     self.calculate(atoms)
    #     return self.results['dielectric']
    
    @staticmethod
    def data_to_atoms_list(data: Data) -> List[Atoms]:
        """
        This helper method takes a 'torch_geometric.data.Data' object containing information about atomic structures
        and converts it into a list of 'ase.Atoms' objects. Each 'Atoms' object represents an atomic structure
        with its associated properties such as positions and cell.
        
        Args:
        - data (Data): A data object containing information about atomic structures.
            
        Returns:
        - List[Atoms]: A list of 'ase.Atoms' objects, each representing an atomic structure
            with positions and associated properties.
        """
        cells = data.cell.numpy()
        
        split_indices = np.cumsum(data.n_atoms)[:-1]
        positions_per_structure = np.split(data.pos.numpy(), split_indices)
        symbols_per_structure = np.split(data.z.numpy(), split_indices)
        
        atoms_list = [Atoms(
                        symbols=symbols_per_structure[i],
                        positions=positions_per_structure[i],
                        cell=Cell(cells[i])) for i in range(len(data.structure_id))]
        for i in range(len(data.structure_id)):
            atoms_list[i].structure_id = data.structure_id[i][0]
        return atoms_list
    
    @staticmethod
    def _load_model(config: dict, rank: str) -> List[BaseModel]:
        """
        This static method loads a model based on the provided configuration.

        Parameters:
        - config (dict): Configuration dictionary containing model and dataset parameters.
        - rank: Rank information for distributed training.

        Returns:
        - model_list: A list of loaded models.
        """
        
        graph_config = config['dataset']['preprocess_params']
        model_config = config['model']
        
        model_list = []
        model_name = model_config["name"]
        logging.info(f'MDLCalculator: setting up {model_name} for calculation')
        # Obtain node, edge, and output dimensions for model initialization   
        for _ in range(model_config["model_ensemble"]): 
            node_dim = graph_config["node_dim"]
            edge_dim = graph_config["edge_dim"]   

            model_cls = registry.get_model_class(model_name)
            model = model_cls(
                    node_dim=node_dim, 
                    edge_dim=edge_dim, 
                    output_dim=1, 
                    cutoff_radius=graph_config["cutoff_radius"], 
                    n_neighbors=graph_config["n_neighbors"], 
                    graph_method=graph_config["edge_calc_method"], 
                    num_offsets=graph_config["num_offsets"], 
                    **model_config
                    )
            model = model.to(rank)
            model_list.append(model)
        
        checkpoints = config['task']["checkpoint_path"].split(',')
        if len(checkpoints) == 0:
            logging.warning("MDLCalculator: No checkpoint.pt file is found, and untrained models are used for prediction.")
        else:
            for i in range(len(checkpoints)):
                try:
                    checkpoint = torch.load(checkpoints[i], map_location=torch.device('cpu'))
                    model_list[i].load_state_dict(checkpoint["state_dict"])
                    logging.info(f'MDLCalculator: weights for model No.{i+1} loaded from {checkpoints[i]}')
                except ValueError:
                    logging.warning(f"MDLCalculator: No checkpoint.pt file is found for model No.{i+1}, and an untrained model is used for prediction.")


        return model_list
