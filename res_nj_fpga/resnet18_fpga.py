# This file is used to generate the custom task for the FPGA implementation of the ResNet18 model.
# Different layers of the ResNet18 model are implemented as custom tasks in this file.

import sys
import math
sys.path.append("../../compiler")
from custom_task import *

N = 8192

def coeff_conv_fpga_optimized_1(id,global_H,global_W,global_kh,global_kw,global_s,global_p,global_Ci,global_Co):   
    global_H,global_W,global_kh,global_kw,global_s,global_p,global_Ci,global_Co = global_H,global_W,global_kh,global_kw,global_s,global_p,global_Ci,global_Co
    level = 1
    Co,tile,numpoly = 0,0,0

    channel_per_poly = 0
    Co = global_Co
    metaCi = global_Ci
    metaH = global_H
    metaW = global_W
    metap = global_p
    metakh = global_kh

    Oc=0 
    tile=1
    kh1=metakh//2
    Sw=metaW+2*metap
    Sh=metaH+2*metap

    for c in range(metaCi):
        Oc = c*(Sh+2*kh1)*(Sw+2*kh1)+(metakh-1)*(Sw+2*kh1)+metakh-1+(Sw+2*kh1)*(Sh-1)+Sw-1
        if Oc < N:
            channel_per_poly +=1
        else:
            break
    if channel_per_poly == 0:
        channel_per_poly = 1
        while True:
            tile+=1
            Sw=((metaW+2*metap)+tile-1)//tile
            Sh=((metaH+2*metap)+tile-1)//tile
            Oc=(metakh-1)*(Sw+2*kh1)+metakh-1+(Sw+2*kh1)*(Sh-1)+Sw-1
            if Oc < N:
                break
    numpoly = (metaCi+channel_per_poly-1)//channel_per_poly

    print(numpoly*tile*tile)
    a_ct = [CiphertextNode(f'a_{i}', level) for i in range(numpoly*tile*tile)]
    w_pt = [PlaintextRingtNode(f'w_{i}') for i in range(Co*numpoly)]
    Co_ct = [CiphertextNode(f'out_{i}',level) for i in range(Co*tile*tile)]

    for co in range(Co):
        for ti in range(tile):
            for tj in range(tile):
                    Co_ct[co*tile*tile+ti*tile+tj] = ct_pt_mult_accumulate(a_ct[ti*tile*numpoly+tj*numpoly:ti*tile*numpoly+tj*numpoly+numpoly],w_pt[co*numpoly:co*numpoly+numpoly])
    
    process_custom_task(
        algo='BFV',
        input_args = [Argument('a', a_ct),Argument('w',w_pt)],  
        output_args = [Argument('out', Co_ct)], 
        output_instruction_path='coeff_resnet18_fpga_'+str(id),
    )


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
    cnt = 0

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
        cnt += 1
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

    print(id, cnt)
    # process_custom_task(
    #     algo='BFV',
    #     input_args = [Argument('w', w), Argument('ac', ac)], 
    #     output_args = [Argument('y', y)], 
    #     output_instruction_path=f'neujeans{id}',
    # )


if __name__ == '__main__':
    # coeff_conv_fpga_optimized_1(10,224,224,7,7,2,3,3,64)
    neujeans_fpga(0,56,56,3,3,1,1,64,64)
    neujeans_fpga(1,56,56,3,3,2,1,64,128)
    neujeans_fpga(2,28,28,3,3,1,1,128,128)
    neujeans_fpga(3,56,56,1,1,2,0,64,128)
    neujeans_fpga(4,28,28,3,3,2,1,128,256)
    neujeans_fpga(5,14,14,3,3,1,1,256,256)
    neujeans_fpga(6,28,28,1,1,2,0,128,256)
    neujeans_fpga(7,14,14,3,3,2,1,256,512)
    neujeans_fpga(8,7,7,3,3,1,1,512,512)
    neujeans_fpga(9,14,14,1,1,2,0,256,512)
    