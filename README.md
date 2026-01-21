# PDI
Calculate Perfusion Disconnectivity 
PDI ©️ 2025 by Philipp J. Koch is licensed under CC BY-NC-SA 4.0  
[https://creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)  

This work is based on the following and distributed under the CC BY-NC-SA 4.0 License:  
Please cite the following when using this code  
- Population based Tractogram:    
[Xiao et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/) (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474320/)      
[This data is available here](https://osf.io/p7syt/) (https://osf.io/p7syt/)     


1. Required repositories  
&nbsp;&nbsp;&nbsp;&nbsp;nibabel  
&nbsp;&nbsp;&nbsp;&nbsp;scipy  
&nbsp;&nbsp;&nbsp;&nbsp;dipy  
&nbsp;&nbsp;&nbsp;&nbsp;antspyx  
&nbsp;&nbsp;&nbsp;&nbsp;pandas  
&nbsp;&nbsp;&nbsp;&nbsp;alternative: activate the given environment (s. 3.)

3. Clone the repository
```bash
git clone https://github.com/phjkoch/PDI.git
cd PDI
```

3. Activate Environment
```bash
conda env create -f environment.yml
conda activate pdi
```

4. Usage

```bash
python PDI.py --help
```
&nbsp;&nbsp;&nbsp;&nbsp;Usage:  
&nbsp;&nbsp;&nbsp;&nbsp;Create_NTDisconn.py ID in_lesion output_dir  

&nbsp;&nbsp;&nbsp;&nbsp;positional arguments:  
&nbsp;&nbsp;&nbsp;&nbsp;ID:                    Subject ID  
&nbsp;&nbsp;&nbsp;&nbsp;input:                 Input folders with DICOMs of Perfusion maps 
&nbsp;&nbsp;&nbsp;&nbsp;output_dir:            Specify output directory
&nbsp;&nbsp;&nbsp;&nbsp;side:                  Specify affected hemisphere

                

5. Output  
&nbsp;&nbsp;&nbsp;In the output_dir a directory named after the ID is created containing  
&nbsp;&nbsp;&nbsp;2. A txt file with 2 millionen entries indicating which streamlines of the HCP-aging tractogram is passing through the individual lesion mask [1] and which are sparsed [0] for the CBF < 30%, Tmax > 6 sec and the respective penumbra



