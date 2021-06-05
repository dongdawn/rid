# CASP13 analysis

## Prerequisite
- Python 3
- Python libraries: NumPy, MDTraj, MSMBuilder

## Initial setup
- Decompress the tgz files   
  tar xzf tICA_model.tgz   
  tar xzf tICA_npz.tgz   
  
## Files in each directory
- scripts: scripts for the tICA analysis
- native/init: the native structures and the initial models
- example: example files
- tICA_model: pickle files for tICA models 
- tICA_npz: tICA coordinates for trajectories used for the analysis described in the paper.   
  You can load the npz file as follows:   
  <pre><code>tICA_crd = np.load(npz_file, allow_pickle=True, encoding='latin1')</pre></code>    
  There are three items in tICA_crd:   
  - tICA_crd['ref_crd']: tICA coordinates for the native structure
  - tICA_crd['ini_crd']: tICA coordinates for the initial model
  - tICA_crd['tICA_crd']: tICA coordinates for the MD simulation trajectories
  
## How to run the script
./scripts/proj_tICA.py --target TARGET --top TOPOLOGY_PDB_FILE --traj TRAJECTORY_FILEs --output OUTPUT.npy
- TARGET: one of the CASP target IDs, R0974s1, R0986s1, R0986s2, R1002-D2
- TOPOLOGY_PDB_FILE: topology file in PDB format for reading trajectory files
- TRAJECTORY_FILEs: trajectory files to get tICA coordinates. You can provide multiple files at a time.
- OUTPUT.npy: mapped tICA coordinates in a NumPy npy file.   
  In the output npy file, it contains a NumPy array or a list with a length of (Number of trajectory files), and each item in the list is an array of tICA coordinates with a shape of (Number of frames, Number of tICs(=5)).

## Example
In the example directory, there is an example script, run.sh, and example files.

## References
- Lim Heo, Collin F. Arbour, and Michael Feig, Driven to Near-Experimental Accuracy by Refinement via Molecular Dynamics Simulations. *Proteins*, (2019) __87__, 1263-1275 [LINK](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25759)
