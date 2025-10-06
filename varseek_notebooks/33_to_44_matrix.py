import numpy as np
import sys

def convert_3x3_to_4x4(input_file, output_file):
    """
    Convert a 3x3 affine transformation matrix to 4x4 format for TransformJ.
    
    Args:
        input_file: Path to input CSV file containing 3x3 matrix
        output_file: Path to output CSV file for 4x4 matrix
    """
    try:
        # Read the 3x3 matrix from CSV
        matrix_3x3 = np.loadtxt(input_file, delimiter=',')
        
        # Validate dimensions
        if matrix_3x3.shape != (3, 3):
            raise ValueError(f"Input matrix must be 3x3, but got shape {matrix_3x3.shape}")
        
        # Create 4x4 matrix
        # 3x3 format:
        # [m00 m01 tx]
        # [m10 m11 ty]
        # [0   0   1 ]
        #
        # 4x4 format for TransformJ:
        # [m00 m01 0  tx]
        # [m10 m11 0  ty]
        # [0   0   1  0 ]
        # [0   0   0  1 ]
        
        matrix_4x4 = np.array([
            [matrix_3x3[0, 0], matrix_3x3[0, 1], 0, matrix_3x3[0, 2]],
            [matrix_3x3[1, 0], matrix_3x3[1, 1], 0, matrix_3x3[1, 2]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Save to CSV
        np.savetxt(output_file, matrix_4x4, delimiter=',', fmt='%.15f')
        
        print(f"Successfully converted 3x3 matrix to 4x4")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print("\n4x4 Matrix:")
        print(matrix_4x4)
        
        return matrix_4x4
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_matrix.py <input_3x3.csv> <output_4x4.csv>")
        print("\nExample:")
        print("  python convert_matrix.py transform_3x3.csv transform_4x4.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_3x3_to_4x4(input_file, output_file)


if __name__ == "__main__":
    main()