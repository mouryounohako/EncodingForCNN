# This file is used to generate the custom task for the FPGA implementation of the ResNet18 model.
# Different layers of the ResNet18 model are implemented as custom tasks in this file.

import sys
sys.path.append("../../../compiler")
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



if __name__ == '__main__':
    # coeff_conv_fpga_optimized_1(10,224,224,7,7,2,3,3,64)
    # coeff_conv_fpga_optimized_1(0,56,56,3,3,1,1,64,64)
    # coeff_conv_fpga_optimized_1(1,56,56,3,3,2,1,64,128)
    # coeff_conv_fpga_optimized_1(2,28,28,3,3,1,1,128,128)
    # coeff_conv_fpga_optimized_1(3,56,56,1,1,2,0,64,128)
    # coeff_conv_fpga_optimized_1(4,28,28,3,3,2,1,128,256)
    # coeff_conv_fpga_optimized_1(5,14,14,3,3,1,1,256,256)
    # coeff_conv_fpga_optimized_1(6,28,28,1,1,2,0,128,256)
    # coeff_conv_fpga_optimized_1(7,14,14,3,3,2,1,256,512)
    # coeff_conv_fpga_optimized_1(8,7,7,3,3,1,1,512,512)
    # coeff_conv_fpga_optimized_1(9,14,14,1,1,2,0,256,512)
    pass