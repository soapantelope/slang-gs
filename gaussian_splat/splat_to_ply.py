import struct
import numpy as np
import math

def read_splat_file(filepath):
    splats_data = []
    
    # Length of each splat
    SPLAT_BYTE_LENGTH = 32

    # Struct format based on the byte layout
    # '<': little-endian
    # 'fff': 3 floats (positions & scales)
    # 'BBBB': 4 unsigned chars (colors & rotations)
    STRUCT_FORMAT = '< fff fff BBBB BBBB' # = 32 bytes

    try:
        with open(filepath, 'rb') as f:
            while True:
                data_bytes = f.read(SPLAT_BYTE_LENGTH)
                
                if not data_bytes or len(data_bytes) < SPLAT_BYTE_LENGTH:
                    break
                
                unpacked_data = struct.unpack(STRUCT_FORMAT, data_bytes)
                
                pos_x, pos_y, pos_z = unpacked_data[0:3]
                
                read_scale_0, read_scale_1, read_scale_2 = unpacked_data[3:6]
                scale_0 = math.log(read_scale_0) if read_scale_0 > 0 else -1e-6
                scale_1 = math.log(read_scale_1) if read_scale_1 > 0 else -1e-6
                scale_2 = math.log(read_scale_2) if read_scale_2 > 0 else -1e-6
                
                color_r_byte, color_g_byte, color_b_byte = unpacked_data[6:9]
                f_dc_0 = (color_r_byte / 255.0 - 0.5) * 2.0 * 1.772196
                f_dc_1 = (color_g_byte / 255.0 - 0.5) * 2.0 * 1.772196
                f_dc_2 = (color_b_byte / 255.0 - 0.5) * 2.0 * 1.772196

                opacity_byte = unpacked_data[9]
                opacity_float_0_1 = opacity_byte / 255.0
                opacity_float_0_1 = max(0.0001, min(0.9999, opacity_float_0_1))
                opacity = math.log(opacity_float_0_1 / (1 - opacity_float_0_1))
                
                rot_0_byte, rot_1_byte, rot_2_byte, rot_3_byte = unpacked_data[10:14]
                rot_0 = (rot_0_byte / 128.0) - 1.0
                rot_1 = (rot_1_byte / 128.0) - 1.0
                rot_2 = (rot_2_byte / 128.0) - 1.0
                rot_3 = (rot_3_byte / 128.0) - 1.0
                
                splat_info = {
                    'x': pos_x,
                    'y': pos_y,
                    'z': pos_z,
                    'opacity': opacity,
                    'rot_0': rot_0,
                    'rot_1': rot_1,
                    'rot_2': rot_2,
                    'rot_3': rot_3,
                    'f_dc_0': f_dc_0,
                    'f_dc_1': f_dc_1,
                    'f_dc_2': f_dc_2,
                    'scale_0': scale_0,
                    'scale_1': scale_1,
                    'scale_2': scale_2,
                }
                splats_data.append(splat_info)
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return splats_data


def write_ply_file(splats_data, output_filepath, format):
    if not splats_data:
        print("No splats found.")
        return

    num_splats = len(splats_data)

    try:
        with open(output_filepath, 'w' if format == "ascii" else 'wb') as f:
            header_lines = [
                    "ply\n",
                    f"format binary_little_endian 1.0\n" if format == "binary" else "format ascii 1.0\n",
                    f"element vertex {num_splats}\n",
                    "property float x\n",
                    "property float y\n",
                    "property float z\n",
                    "property float opacity\n",
                    "property float rot_0\n",
                    "property float rot_1\n",
                    "property float rot_2\n",
                    "property float rot_3\n",
                    "property float f_dc_0\n",
                    "property float f_dc_1\n",
                    "property float f_dc_2\n",
                    "property float scale_0\n",
                    "property float scale_1\n",
                    "property float scale_2\n",
                    "end_header\n"
            ]        
            if format == "binary":
                for line in header_lines:
                    f.write(line.encode('utf-8'))
                for splat in splats_data:
                    data = struct.pack('<3f1f4f3f3f', 
                                       splat['x'], splat['y'], splat['z'],
                                       splat['opacity'],
                                       splat['rot_0'], splat['rot_1'], splat['rot_2'], splat['rot_3'],
                                       splat['f_dc_0'], splat['f_dc_1'], splat['f_dc_2'],
                                       splat['scale_0'], splat['scale_1'], splat['scale_2']
                                       )
                    f.write(data)
            else:
                for line in header_lines:
                    f.write(line)
                for splat in splats_data:
                    f.write(
                        f"{splat['x']} {splat['y']} {splat['z']} "
                        f"{splat['opacity']} "
                        f"{splat['rot_0']} {splat['rot_1']} {splat['rot_2']} {splat['rot_3']} "
                        f"{splat['f_dc_0']} {splat['f_dc_1']} {splat['f_dc_2']} "
                        f"{splat['scale_0']} {splat['scale_1']} {splat['scale_2']}\n")

        print(f"Successfully wrote {num_splats} splats to '{output_filepath}'.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a .splat file to a .ply file.")
    parser.add_argument("-i", "--input", help="Input .splat file")
    parser.add_argument("-o", "--output", help="Output .ply file")
    parser.add_argument("-f", "--format", help="Format: ascii, binary (binary_little_endian)", default="binary")
    args = parser.parse_args()

    read_splats = read_splat_file(args.input)
    write_ply_file(read_splats, args.output, args.format) 