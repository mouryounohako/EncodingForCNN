import sys
import math
sys.path.append("../../compiler")
from custom_task import *


def bolt_fpga():    
    level = 1
    N, d0, d1, d2, t, inrot = 4096, 64, 1024, 64, 64, 8  # d0 /= 2
    # inrot = pow(2, math.floor(0.5 * math.log2(d2 * t / d1)))
    # if (d1 * inrot + d2 * t / inrot > 2 * d1 * inrot + d2 * t / (2 * inrot)):
    #     inrot *= 2

    w = [PlaintextMulNode(f'w_{_}', level) for _ in range(d1 * d2 // t)]
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(d1 // t)]
    y = []
    
    ac_rot = []
    for i in range(d1 // t):
        ac_rot.append(ac[i])
    for i in range(inrot - 1):
        for j in range(d1 // t):
            ac_rot.append(rotate_cols(ac_rot[i * (d1 // t) + j], d0, f"rot_{i}_{j}")[0])
    int_tmp = []
    for i in range(len(w)):
        int_tmp.append(mult(ac_rot[(inrot - 1 - i % t % inrot) * d1 // t + i // d2], w[i], f"int_{i}"))
    for i in range(1, d1 // t):
        for j in range(d2):
            int_tmp[j] = add(int_tmp[j], int_tmp[j + i * d2])
    for i in range(d2 // t):
        # for j in range(t):
        #     int_tmp.append(ct_pt_mult_accumulate([ac_rot[(inrot - 1 - j % inrot) * (d1 // t) + k] for k in range(d1 // t)], [w[k * d2 + i * t + j] for k in range(d1 // t)]))
        for j in range(t):
            if (j % inrot != 0):
                int_tmp[i * t + j - j % inrot] = add(int_tmp[i * t + j - j % inrot], int_tmp[i * t + j]) # may be merged if needed later
        y.append(int_tmp[i * t])
        for j in range(1, t // inrot):
            y[i] = rotate_cols(y[i], d0 * inrot)[0]
            y[i] = add(y[i], int_tmp[i * t + j * inrot])

    process_custom_task(
        algo='BFV',
        input_args = [Argument('w', w), Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path='bolt',
    )


if __name__ == '__main__':
    bolt_fpga()
