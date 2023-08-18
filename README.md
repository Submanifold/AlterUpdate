# Alternately denoising and reconstructing unoriented point sets

The implementation of ["Alternately denoising and reconstructing unoriented point sets"](https://arxiv.org/abs/2305.00391).
In this work, we propose a new strategy to bridge point cloud denoising and surface reconstruction by alternately updating the denoised point clouds and the reconstructed surfaces.

The main function is in "alterupdate.py". The code runs in Windows and requires trimesh, rtree and numpy, which can be easily installed through Python‘s pip. Our experimental environment is "python 3.6.5, numpy==1.19.5, trimesh==3.9.0 and rtree==0.9.7".
The folder "./utils" contains several basic I/O operations, which are implemented by [Points2Surf]( https://github.com/ErlerPhilipp/points2surf).

The program takes a noisy point cloud as input and generates the denoised point cloud and the reconstructed mesh. The default input folder is "./input", and the default output folder is "./output".
The intermediate results in each iteration be generated in the "./intermediate" folder.

There are several parameters in the program. The specific descriptions of the parameters are in function "parse_arguments (line 8)" of "alterupdate.py".

Here are several examples, the commands can be executed in the root folder of this project.

```bash
python alterupdate.py --xyz_name horse
python alterupdate.py --xyz_name hand
python alterupdate.py --xyz_name 3DBenchy
python alterupdate.py --xyz_name armadillo_large_noise
python alterupdate.py --xyz_name xyzrgb_statuette --poisson_weight 0.5
python alterupdate.py --xyz_name fandisk_noisy --specific_param_edge True --c 0.11 --sigma 0.05
```

The execution of "alterupdate.py" relies on two executable files, "ipsr.exe" and "edges_value.exe", which are located in the root path of this project.
"ipsr.exe" is compiled using the VS solution in the folder "./iPSR". We slightly modified the I/O of original [iPSR](https://github.com/houfei0801/ipsr) implementation to support xyz inputs and add a new parameter "--random_init".

The lambda-projection scheme of our method relies on the feature detect module "edge_value.exe", which is complied using [CGAL](www.cgal.org). 
We slightly modified the CGAL example "Point_set_processing_3/edges_example.cpp" to generate the specific sharpness ratio of each point. 


## Citation
To be updated.

