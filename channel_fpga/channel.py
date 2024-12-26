import sys
import math
sys.path.append("../../compiler")
from custom_task import *


def channel_fpga(id, H, W, kh, kw, s, p, Ci, Co):    
    level, N = 1, 8192
    H += 2 * p
    W += 2 * p
    C = N // (H * W)
    c = 1
    while (c <= C):
      c *= 2
    C = c // 2
    ci, co = Ci // C + (Ci % C != 0), Co // C + (Co % C != 0)
    automap = {}
    automap[1] = [2048]
    automap[2] = [2048, 1024]
    automap[4] = [2048, 1024, 512]
    automap[8] = [2048, 1024, 512, 256]
    automap[16] = [2048, 1024, 512, 256, 128]
    automap[32] = [1024, 512, 256, 128, 64]
    automap[64] = [512, 256, 128, 64, 32]
    automap[128] = [1024, 512, 256, 128, 64, 32, 16]
    automap[256] = [512, 256, 64, 32, 16, 8]
    automap[512] = [256, 128, 16, 8, 4]
    automap[1024] = [2048, 1024, 256, 64, 4, 2]
    automap[2048] = [1]
    rot_cnt = 0

    w = [PlaintextRingtNode(f'w_{_}') for _ in range(ci * Co)]
    r = [PlaintextRingtNode(f'r_{_}') for _ in range(int(math.log2(C)))]  # shift 1, 2, ..., C / 2 slots
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(ci)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(co)]
    
    int_ct = []
    if (len(ac) > 1):
        for i in range(Co):
            int_ct.append(ct_pt_mult_accumulate(ac, [w[j * Co + i] for j in range(ci)]))
    else:
        for i in range(Co):
            int_ct.append(mult(ac[0], w[i], f'int_{i}'))

    for i in range(int(math.log2(C))):
        fold = C // (2 ** (i + 1))
        print(fold)
        for j in range(co):
            for k in range(fold):
                tmp1 = int_ct[j * C + k]
                tmp2 = int_ct[j * C + k + fold]
                for l in range(0, len(automap[fold])):
                    tmp1 = rotate(tmp1, automap[fold][l])
                    tmp2 = rotate(tmp2, automap[fold][l])
                rot_cnt += 2
                int_ct[j * C + k] = add(int_ct[j * C + k], tmp1)
                int_ct[j * C + k + fold] = add(int_ct[j * C + k + fold], tmp2)
                shift = mult(int_ct[j * C + k + fold], r[int(math.log2(C)) - i - 1])
                int_ct[j * C + k] = add(int_ct[j * C + k], shift)
            y[j] = int_ct[j * C]

    # process_custom_task(
    #     algo='BFV',
    #     input_args = [Argument('w', w), Argument('ac', ac), Argument('r', r)], 
    #     output_args = [Argument('y', y)], 
    #     output_instruction_path=f'channel_{id}',
    # )


def channel_fpga_optimized(id, H, W, kh, kw, s, p, Ci, Co):    
    level, N = 1, 8192
    H += 2 * p
    W += 2 * p
    d = 2 ** math.ceil(math.log2(H * W))
    C = N // d
    ci, co = Ci // C + (Ci % C != 0), Co // C + (Co % C != 0)
    inrot = math.sqrt(C)
    inrot = int(inrot * math.sqrt(2)) if inrot != math.floor(inrot) else int(inrot)
    automap = [2048, 3072, 3584, 3840, 3968, 1984, 992]

    w = [PlaintextRingtNode(f'w_{_}') for _ in range(ci * co * C)]
    ac = [CiphertextNode(f'ac_{_}', level) for _ in range(ci)]
    y = [CiphertextNode(f'y_{_}', level) for _ in range(co)]
    
    ac_rot = []
    for i in range(ci):
        ac_rot.append(ac[i])
    for i in range(inrot - 1):
        for j in range(ci):
            if C > 1:
                ac_rot.append(rotate(ac_rot[i * ci + j], automap[int(math.log2(C) - 1)], f"rot_{i}_{j}"))

    int_tmp = []
    # len(w) = ci * co * C
    # 先加到第一个co * C里
    # 剩下co个1到C，再把每个inrot加起来
    for i in range(co * C // inrot):
        ac_ls, w_ls = [], []
        for j in range(inrot):
            for k in range(ci):
                ac_ls.append(ac_rot[j * ci + k])
                w_ls.append(w[k * co * C + i * inrot + j])
        int_tmp.append(ct_pt_mult_accumulate(ac_ls, w_ls))

    for i in range(co):
        y[i] = int_tmp[(i + 1) * C // inrot - 1]
        for j in range(C // inrot - 1):
            if C > 1:
                y[i] = rotate(y[i], (automap[int(math.log2(C) - 1)] * inrot) % 4096)
            y[i] = add(y[i], int_tmp[(i + 1) * C // inrot - j - 2])

    process_custom_task(
        algo='BFV',
        input_args = [Argument('w', w), Argument('ac', ac)], 
        output_args = [Argument('y', y)], 
        output_instruction_path=f'channel_bsgs_{id}',
    )


if __name__ == '__main__':
    channel_fpga_optimized(0,9,9,3,3,1,1,64,64)
