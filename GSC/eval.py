# coding=utf-8
import os
from evaluation import eval_DECL

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # DECL-SGRAF.
    avg_SGRAF = False
    
    data_path= '/remote-home/share/zhaozh/NC_Datasets/data'
    vocab_path= '/remote-home/share/zhaozh/NC_Datasets/vocab'
    ns = [0]
    for n in ns:
        print(f"\n============================>f30k noise:{n}")
        # checkpoint_paths = [
        #     f'./f30K_SAF_noise{n}_best.tar',
        #     f'./f30K_SGR_noise{n}_best.tar']
        checkpoint_paths = [f'/remote-home/zhaozh/NC/ELCL_TOPO/runs/f30k_gamma/r_corr_0/gamma_01/checkpoint_dir/model_best.pth.tar']
        eval_DECL(checkpoint_paths, avg_SGRAF, data_path=data_path, vocab_path=vocab_path)

    # for n in ns:
    #     print(f"\n============================>coco noise:{n}")
    #     # checkpoint_paths = [
    #     #     f'./coco_SAF_noise{n}_best.tar',
    #     #     f'./coco_SGR_noise{n}_best.tar']
    #     checkpoint_paths = [f'/remote-home/zhaozh/NC/DECL/runs/coco/r_corr_{n}/checkpoint_dir/model_best.pth.tar']
    #     eval_DECL(checkpoint_paths, avg_SGRAF, data_path=data_path, vocab_path=vocab_path)

    # print(f"\n============================>cc152k")
    # eval_DECL(['./cc152k_SAF_best.tar', './cc152k_SGR_best.tar'], avg_SGRAF=avg_SGRAF)
