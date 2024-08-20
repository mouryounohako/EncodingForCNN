# This file is used to generate the custom task for the FPGA implementation of the ResNet18 model.
# Different layers of the ResNet18 model are implemented as custom tasks in this file.

import sys
sys.path.append("../../compiler")
from custom_task import *

N = 8192

def test_rot(n, r):
    level = 1
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(n)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(n)]
    
    for i in range(n):
        y[i] = rotate(ac[i], r)

    process_custom_task(
        algo='BFV',
        input_args = [Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path='test_rotation',
    )


def test_mult(n):
    level = 1
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(n)]
    w = [PlaintextRingtNode(f'w_{_}') for _ in range(n)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(n)]
    
    for i in range(n):
        y[i] = mult(ac[i], w[i])

    process_custom_task(
        algo='BFV',
        input_args = [Argument('w', w), Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path='test_mult',
    )


def test_mult_add(n):
    level = 1
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(n)]
    w = [PlaintextRingtNode(f'w_{_}') for _ in range(n)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(n)]
    
    for i in range(n):
        y[i] = mult(ac[i], w[i])

    process_custom_task(
        algo='BFV',
        input_args = [Argument('w', w), Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path='test_mult',
    )


if __name__ == '__main__':
    test_mult(500)
    # rot(500, 1) = 0.02047
    # rot(500, 64) = 0.02023
    # mult(500) = 0.02415
    """
    multiplication time: 
    0 2048 * 4
    1 4096
    2 2048 * 3
    3 4096
    4 4096
    5 2048 * 3
    6 4096
    7 4096
    8 8192 * 3
    9 4096
    """
