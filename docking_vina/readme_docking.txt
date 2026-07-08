
Overview

1. Code Mechanism and Representation
2. Assumptions and Limitations
4. TODO
5. References 

Code Mechanism and Representation

1. Clean and prepare mutation protein using from PDB 
2. Generate mutation protein from alphafold server 
3. Dock all known ligands and get average score for docking between them
4. To use docking score as y target during training 
 
Steps to clean PDB protein using MGLTools 

1. Remove ligand
2. Remove water
3. Check and repair missing atoms
4. Add and spread charge
5. Draw gridbox and retrieve coordinates

TODO

1. In code

Assumptions

1. Docking scores reflect on mutation proteins for a given ligand
2. Averaging reduces uncertainty and errors in docking scores 

Limitations

1. Not all direct PDB proteins are available, some had to be manually predicted
2. Subject to user error during general protein preparation and gridbox steps 
3. All water molecules removed in target protein so that both versions are same
4. Protein Charges were spread even, pH not verified, tautomers were ignored.  


References/pip/libraries/LLM

1. vina, autodock MGLTools, UCSF Chimera, Pymol, alphafold, RDkit
2. Claude AI assited with reading and saving files during code development

Journal References

1.	Abramson J, Adler J, Dunger J, Evans R, Green T, Pritzel A, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. 2024;630(June). 
2.	Joshi A, Kaushik V. Insights of Molecular Docking in Autodock-Vina : A Practical Approach. 2021;9:1–6. 

