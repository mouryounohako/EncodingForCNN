import sys
import math
sys.path.append("../../compiler")
from custom_task import *


def neujeans_fpga(id, H, W, kh, kw, s, p, Ci, Co):    
    level, N = 1, 8192
    H += 2 * p
    W += 2 * p
    H, W = 2 ** math.ceil(math.log2(H)), 2 ** math.ceil(math.log2(W))
    C = N // (2 * H * W)
    Co = Co // 2
    ci, co = Ci // C + (Ci % C != 0), Co // C + (Co % C != 0)
    inrot = math.sqrt(C)
    inrot = int(inrot * math.sqrt(2)) if inrot != math.floor(inrot) else int(inrot)

    #  ci = d1 / t
    #  co = d2 / t
    #  C = t
    #  H * W = d0
    w = [PlaintextMulNode(f'w_{_}', level) for _ in range(ci * co * C)]
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(ci)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(co)]
    
    ac_rot = []
    for i in range(ci):
        ac_rot.append(ac[i])
    for i in range(inrot - 1):
        for j in range(ci):
            ac_rot.append(rotate(ac_rot[i * ci + j], H * W, f"rot_{i}_{j}"))
    int_tmp = []
    for i in range(len(w)):
        int_tmp.append(mult(ac_rot[(inrot - 1 - i % inrot) * ci + i // (co * C)], w[i], f"int_{i}"))
    for i in range(1, ci):
        for j in range(co * C):
            int_tmp[j] = add(int_tmp[j], int_tmp[j + i * co * C])
    for i in range(co):
        # for j in range(t):
        #     int_tmp.append(ct_pt_mult_accumulate([ac_rot[(inrot - 1 - j % inrot) * (d1 // t) + k] for k in range(d1 // t)], [w[k * d2 + i * t + j] for k in range(d1 // t)]))
        for j in range(C):
            if (j % inrot != 0):
                int_tmp[i * C + j - j % inrot] = add(int_tmp[i * C + j - j % inrot], int_tmp[i * C + j]) # may be merged if needed later
        y[i] = int_tmp[i * C]
        for j in range(C // inrot - 1):
            y[i] = rotate(y[i], H * W * inrot)
            y[i] = add(y[i], int_tmp[i * C + j * inrot + inrot])

    process_custom_task(
        algo='BFV',
        input_args = [Argument('w', w), Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path=f'neujeans_{id}',
    )


if __name__ == '__main__':
    neujeans_fpga(0,56,56,3,3,1,1,64,64)
