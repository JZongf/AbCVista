CISRR 
version 1.0
(c) 2010
Institute of Biophysics, Chinese Academy of Sciences

Description
CISRR is a user friendly program for  accurate prediction of protein side chains and effective elimination of their inner atomic clashes. 
Most of the state-of-the-art side-chain modeling methods are based on discrete rigid side-chain rotamers. The use of discrete rigid rotamers could inherently lead to atomic clashes and prevent the accurate modeling of protein side chains. To overcome this issue CISRR couples a novel clash-detection guided iterative search algorithm (CIS) with rotamer relaxation (RR), which  removes atomic clashes effectively and achieves high accuracy (about 86% and 76% for Chi1 and Chi1&2, respectively, within 40° cutoff).
CISRR could be a very useful tool for subsequent analysis and refinement of protein structures. 
CISRR is free for non-commercial users. Other users please contact us.
Please Cite
Yang Cao, Lin Song, Zhichao Miao, Yun Hu, Liqing Tian, Taijiao Jiang. Improved side-chain modeling by coupling clash-detection guided iterative search with rotamer relaxation. Bioinformatics, 2011 Mar 15;27(6):785-90

Install
CISRR is a command-line application under Linux.
Extract the CISRR.tar.gz and place the CISRR folder anywhere in your computer whithout modifying the subdirectories and files.
Then you can run the software (<program path>/bin/CISRR).
Usage
:
-i [Input PDB File]
The protein whose side-chains will be packed. It should be a PDB file which contains
 backbone atom records (N, CA, C, O).
 Alternative atoms (e.g., alt_loc of "B") and OXT atoms are ignored.
-o [Output PDB file with predicted side-chain information]
 
Example:
~$ ../bin/CISRR -i 1AGIcln_main.pdb -o 1AGIcln_sp.pdb 
***********************CISRR***********************
Hello! CISRR is ready for your side-chain modeling.
         Institute of Biophysics, CAS
           Mon Oct 18 16:55:18 2010
***********************CY2010**********************
Protein Atom: 1013  Res: 124
Defective Residues 115
Iteration Round 1 Score: 71.4013
Iteration Round 2 Score: -157.9404
Iteration Round 3 Score: -176.2625
Iteration Round 4 Score: -176.7581
Iteration Round 5 Score: -186.7023
Iteration Round 6 Score: -191.0497
Success! Final Score: -191.1450
Run Time: 6.920 sec.

Options
-l [ligand file (mol2 format)]: Ligand Restriction for side-chain modeling
-H [add hydrogen atoms]: The hydrogen atoms will be added according to Charmm22 standard topologies.
-h [help]: Help information;
-m [mutation information]: The format after -m is [chain id] [sequence number] [orginal residue name (3 letter)] [new residue name (3 letter)]
Multiple mutations are supported by adding more -m [mutation info]
-F [Fast packing]: The program will bypass rotamer relaxation and is about 4~5 times faster. (1-2% less accurate) 


History
1. Fixed the output sequence of CG1 and CD1(ILE).  2011.4.


