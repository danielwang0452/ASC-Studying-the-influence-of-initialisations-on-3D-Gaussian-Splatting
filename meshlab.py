import numpy as np
from plyfile import PlyData, PlyElement
import os

from plyfile import PlyData


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def retrieve_elements_from_ply(ply_data):
    # Access vertex data
    vertex_data = ply_data['vertex'].data

    # Extract the specific fields you are interested in
    # Position (x, y, z)
    xyz = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    # Normals (nx, ny, nz)
    normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T

    # f_dc fields (assuming they represent color or other attributes)
    f_dc = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T

    # f_rest fields (additional attributes, there are 45 in total)
    f_rest = np.vstack([vertex_data[f'f_rest_{i}'] for i in range(45)]).T

    # Opacity
    opacity = vertex_data['opacity']

    # Scale (scale_0, scale_1, scale_2)
    scale = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T

    # Rotation (rot_0, rot_1, rot_2, rot_3)
    rotation = np.vstack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T

    # Now you have all the data in the respective numpy arrays.
    return xyz, normals, f_dc, f_rest, opacity, scale, rotation


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def save_xyz_rgb_as_ply(xyz, rgb, output_path):
    # Ensure RGB values are in the range 0-255
    rgb = (rgb * 255).astype(np.uint8)

    # Create the dtype for the PLY format with positions and color attributes (RGB)
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # Prepare the elements combining positions (xyz) and colors (rgb)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    # Describe elements as 'vertex' for the PLY format
    el = PlyElement.describe(elements, 'vertex')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the PLY data to the specified path
    PlyData([el]).write(output_path)
    print(f"PLY file saved to {output_path}")


def gaussianply_to_meshlabply(gaussian_ply_path, save_path):
    ply_data = PlyData.read(gaussian_ply_path)
    xyz, normals, f_dc, f_rest, opacity, scale, rotation = retrieve_elements_from_ply(ply_data)
    rgb = SH2RGB(f_dc)
    #print(rgb.shape)
    #save_xyz_rgb_as_ply(xyz, rgb, save_path)
    storePly(save_path, xyz, 255*rgb)

gaussian_ply_path = '/Users/danielwang/PycharmProjects/gaussian-splatting-test/scene_checkpoints/initialisation.ply'
save_path = '/Users/danielwang/PycharmProjects/gaussian-splatting-test/initialisations/warp_select/bicycle_warp_1M.ply'
gaussianply_to_meshlabply(gaussian_ply_path, save_path)