import trimesh
import os
import numpy as np
from utils import point_cloud
import trimesh.sample
import argparse


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz_name', type=str, default='xyzrgb_statuette', help='the name of the input xyz file (without suffix)')
    parser.add_argument('--input_dir', type=str, default='./input', help='the folder of input files')
    parser.add_argument('--mediate_dir', type=str, default='./intermediate', help='the folder of intermediate files')
    parser.add_argument('--out_dir', type=str, default='./output', help='the folder of results')
    parser.add_argument('--poisson_weight', type=float, default=1.0, help='point weight of Screen Poisson')
    parser.add_argument('--d_min', type=int, default=6, help='min depth in adaptive depth selection')
    parser.add_argument('--d_max', type=int, default=8, help='max depth in adaptive depth selection')
    parser.add_argument('--depth_list', type=list, default=[[6,6,7,7,8],[7,7,8,8,8],[8,8,8,8,8]], help='depth list for all d0s')
    parser.add_argument('--d_sharp', type=int, default=8, help='threshold for sharp features')
    parser.add_argument('--specific_param_edge', type=bool, default=False, help='whether to use specific paramters in feature detection')
    parser.add_argument('--normal_variation_thre', type=float, default=0.7, help='threshold of normal variations in adaptive depth selection')
    parser.add_argument('--R', type=float, default=0.2, help='R in edge detection')
    parser.add_argument('--r', type=float, default=0.05, help='r in edge detection')
    parser.add_argument('--c', type=float, default=0.11, help='c in lambda projection')
    parser.add_argument('--sigma', type=float, default=0.05, help='sigma in lambda projection')

    opt = parser.parse_args(args=args)

    return opt

def projection(xyz, mesh):
    projected_points, _, triangle_id = trimesh.proximity.closest_point(mesh, xyz)
    projected_normals = mesh.face_normals[triangle_id]
    return projected_points, projected_normals

def coeff_edge_array(a, sigma2, thre2):
    return np.exp(-(np.power(np.clip(np.abs(a) - thre2, 0, 10), 2)) / (np.power(sigma2, 2)))


def recon_and_denoise(eval_opt):
    xyz_name = eval_opt.xyz_name
    in_xyz_path = os.path.join(eval_opt.input_dir, xyz_name + ".xyz")
    final_output_xyz_path = os.path.join(eval_opt.out_dir, xyz_name + ".xyz")
    final_output_ply_path = os.path.join(eval_opt.out_dir, xyz_name + ".ply")

    resample_iter = 5

    xyz_data = point_cloud.load_xyz(in_xyz_path)
    max_x = np.max(xyz_data[:, 0])
    min_x = np.min(xyz_data[:, 0])
    max_y = np.max(xyz_data[:, 1])
    min_y = np.min(xyz_data[:, 1])
    max_z = np.max(xyz_data[:, 2])
    min_z = np.min(xyz_data[:, 2])
    center_pc = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2])
    max_length_coord = np.max(np.array(
        [max_x - min_x, max_y - min_y, max_z - min_z]))
    xyz_data[:, :3] = (xyz_data[:, :3] - center_pc) / max_length_coord
    normalized_xyz_path = os.path.join(eval_opt.mediate_dir, xyz_name + "_normalized.xyz")

    if (xyz_data.shape[1] > 3):
        point_cloud.write_xyz(normalized_xyz_path, points=xyz_data[:, :3], normals=xyz_data[:, 3:])
    else:
        point_cloud.write_xyz(normalized_xyz_path, points=xyz_data[:, :3])

    in_xyz_path = normalized_xyz_path
    num_point_total = xyz_data.shape[0]

    R_edge = eval_opt.R
    r_edge = eval_opt.r

    pointWeight = eval_opt.poisson_weight
    normal_variation_thre = eval_opt.normal_variation_thre
    d_max = eval_opt.d_max
    d_min = eval_opt.d_min
    first_depth = d_max
    Poisson_depth = d_max

    adaptive_depth_flag = True

    depth_list = eval_opt.depth_list

    if (len(depth_list) != (d_max - d_min + 1)):
        print("Generate depth list automatically.")
        depth_list = []
        for d in range(d_min, d_max + 1):
            d_all_steps = []
            for s in range(resample_iter):
                d_all_steps.append(int((d_max - d) * s / 5.0 + d))
            d_all_steps[-1] = d_max
            depth_list.append(d_all_steps)

    for i in range(resample_iter):

        xyz_mesh = trimesh.load(in_xyz_path)
        xyz = xyz_mesh.vertices

        ply_out_name = os.path.join(eval_opt.mediate_dir, xyz_name + "_" + str(i) + "ply.ply")
        if i == 0 and adaptive_depth_flag == True:

            d = d_max
            while True:
                depth_d_name = os.path.join(eval_opt.mediate_dir, xyz_name + "_depth" + str(d) + "ply.ply")
                depth_d_iter_txt_name = os.path.join(eval_opt.mediate_dir, xyz_name + "_depth" + str(d) + "ply_iter.txt")
                command_str = "ipsr.exe --in " + in_xyz_path + " --out " + depth_d_name + " --pointWeight " + str(
                    pointWeight) + " --depth " + str(d) + " --random_init 1" + "\n"
                os.system(command_str)
                iter_txt = open(depth_d_iter_txt_name, 'r')
                lines_list = iter_txt.readlines()
                num_iter = int(lines_list[0])
                if num_iter == 30:
                    avg_variation = 0
                    count_variation = 0
                    normals_variations = [float(line) for line in lines_list[1:]]
                    for normals_variation in normals_variations:
                        if (normals_variation > 0 and normals_variation < 2):
                            avg_variation = avg_variation + normals_variation
                            count_variation = count_variation + 1
                    avg_variation = avg_variation / count_variation
                    print("avg variations of last five iterations: " + str(avg_variation))
                    if avg_variation < normal_variation_thre or d == d_min:
                        first_depth = d
                        if (os.path.exists(ply_out_name)):
                            os.replace(depth_d_name, ply_out_name)
                        else:
                            os.rename(depth_d_name, ply_out_name)
                        break
                    else:
                        d = d - 1
                else:
                    first_depth = d
                    if (os.path.exists(ply_out_name)):
                        os.replace(depth_d_name, ply_out_name)
                    else:
                        os.rename(depth_d_name, ply_out_name)
                    break
        else:
            if i >= 1:
                command_str = "ipsr.exe --in " + in_xyz_path + " --out " + ply_out_name + " --pointWeight " + str(
                    pointWeight) + " --depth " + str(Poisson_depth) + " --random_init 0" + "\n"
            else:
                command_str = "ipsr.exe --in " + in_xyz_path + " --out " + ply_out_name + " --pointWeight " + str(
                    pointWeight) + " --depth " + str(Poisson_depth) + " --random_init 1" + "\n"
            print(command_str)
            os.system(command_str)

        depth_list_index = first_depth - d_min
        d_all_steps = depth_list[depth_list_index]
        if (i >= 1):
            Poisson_depth = d_all_steps[i - 1]

        coeff = np.ones(xyz_data.shape[0]) * 0.5
        edge_path = in_xyz_path[:-4] + ".txt"
        if Poisson_depth >= eval_opt.d_sharp:
            command_str_edge = "edges_value.exe " + in_xyz_path + " " + edge_path + " " + str(R_edge) + " " + str(
            r_edge) + "\n"
            print(command_str_edge)
            os.system(command_str_edge)

            sample_edge_detect = point_cloud.load_xyz(edge_path)
            sample_edge_detect = sample_edge_detect[:, 3]

            if eval_opt.specific_param_edge:
                thre2 = eval_opt.c
                sigma2 = eval_opt.sigma
            else:
                top_edge_amount = int(num_point_total / 10.0) + 1
                sorted_edge = np.sort(sample_edge_detect)
                edge_thre = sorted_edge[-top_edge_amount]
                thre2 = edge_thre
                sigma2 = edge_thre / 2.0
            coeff = coeff_edge_array(sample_edge_detect, sigma2, thre2)

        mesh = trimesh.load(ply_out_name)
        projected_points, projected_normals = projection(xyz, mesh)
        if Poisson_depth >= eval_opt.d_sharp:
            r = coeff * 0.9 + 0.1
        else:
            r = np.ones(xyz_data.shape[0]) * 0.5

        new_points = np.zeros((num_point_total, 6)).astype(np.float32)
        x = xyz[:, :3]
        d = projected_points - x
        x = x + np.multiply(np.repeat(np.expand_dims(r, axis=1), 3, axis=1), d)
        new_points[:, :3] = x
        new_points[:, 3:] = projected_normals

        xyz_file_out = os.path.join(eval_opt.mediate_dir, xyz_name + "_" + str(i) + "xyz.xyz")
        point_cloud.write_xyz(file_path=xyz_file_out, points=new_points[:, :3], normals=new_points[:, 3:])
        in_xyz_path = xyz_file_out

        if (i == resample_iter - 1):
            new_points[:, :3] = new_points[:, :3] * max_length_coord + center_pc
            point_cloud.write_xyz(file_path=final_output_xyz_path, points=new_points[:, :3], normals=new_points[:, 3:])
            point_weight_last_recon = 1.0
            final_recon_command_str = "ipsr.exe --in " + final_output_xyz_path + " --out " + final_output_ply_path + " --pointWeight " + str(
                point_weight_last_recon) + " --depth " + str(d_all_steps[-1]) + " --random_init 0" + "\n"
            os.system(final_recon_command_str)
            os.remove(os.path.join(eval_opt.out_dir, xyz_name + "_iter.txt"))
            break


if __name__ == "__main__":

    eval_opt = parse_arguments()
    recon_and_denoise(eval_opt)


