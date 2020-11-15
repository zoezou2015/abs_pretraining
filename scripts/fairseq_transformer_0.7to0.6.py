

import torch, argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_ckpt', default='checkpoint65.pt')
    parser.add_argument('--new_ckpt', default='checkpoint65.0.6.pt')

    return parser.parse_args()


def to_v6(old_ckpt_file, new_ckpt_file):
    ckpt = torch.load(old_ckpt_file)
    state_dict = ckpt['model']
    encoder_ln_map = {'self_attn_layer_norm': 'layer_norms.0', 'final_layer_norm': 'layer_norms.1'}
    decoder_ln_map = {'self_attn_layer_norm': 'layer_norms.0', 'encoder_attn_layer_norm': 'layer_norms.1', 'final_layer_norm': 'layer_norms.2'}
    rep_key_map = []
    for k in state_dict.keys():
        if 'layer_norm' in k:
            if k.startswith('encoder.'):
                for v7, v6 in encoder_ln_map.items():
                    if v7 in k:
                        v6k = k.replace(v7, v6)
                        print(v6k)
                        rep_key_map.append( (k, v6k) )
            elif k.startswith('decoder.'):
                for v7, v6 in decoder_ln_map.items():
                    if v7 in k:
                        v6k = k.replace(v7, v6)
                        print(v6k)
                        rep_key_map.append( (k, v6k) )

    for k, v6k in rep_key_map:
        state_dict[v6k] = state_dict[k]
        del state_dict[k]

    ckpt['model'] = state_dict
    print('************************************************')
    print(state_dict.keys())
    torch.save(ckpt, new_ckpt_file)


if __name__ == '__main__':
    args = get_args()
    print(args)
    to_v6(args.old_ckpt, args.new_ckpt)



