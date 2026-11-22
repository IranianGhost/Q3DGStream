"""Decode Publio's sparse PQAT checkpoints after complete-file QUIC reception.

The wire checkpoint is kept intact. Unretained parameter blocks reconstruct to
zero; retained block IDs are addresses, not transmission priorities. INT2 uses
per-block affine zero points, unlike signed-offset INT4 and symmetric INT8.
"""
import math
import torch


FORMATS={2:'block_affine_int2_u0123_v1',4:'offset_signed_int4_packed',
         8:'symmetric_sparse_blockwise_int8'}
KEYS={2:'model.params_sparse_qint2_packed',4:'model.params_sparse_qint4_packed',
      8:'model.params_sparse_qint8'}


def decode_sparse_checkpoint(state):
    """Return a dense FP32 state for known sparse formats, or pass dense input through."""
    if not isinstance(state,dict):
        return state
    formats=[bits for bits,key in KEYS.items() if key in state]
    if not formats:
        return state
    if len(formats)!=1 or 'model.params' in state:
        raise ValueError('ambiguous NTC parameter encoding')
    bits=formats[0]
    prefix=f'model.params_sparse_int{bits}_'
    try:
        if state[prefix+'quantization']!=FORMATS[bits]:
            raise ValueError('unsupported sparse quantization')
        block=int(state[prefix+'block_size'].item())
        total=int(state[prefix+'total_blocks'].item())
        kept=int(state[prefix+'keep_blocks'].item())
        shape=tuple(int(x) for x in state[prefix+'shape'].reshape(-1).tolist())
        if block!=64 or not shape or any(x<1 for x in shape):
            raise ValueError('invalid sparse shape/block size')
        numel=math.prod(shape)
        if not 0<kept<=total or not 0<numel<=32_000_000 or numel!=total*block:
            raise ValueError('invalid sparse parameter/block count')
        ids=state[prefix+'block_indices'].detach().cpu().reshape(-1)
        scales=state[prefix+'scales'].detach().cpu().float().reshape(-1)
        if ids.dtype not in (torch.int32,torch.int64):
            raise ValueError('block indices must be integers')
        if ids.numel()!=kept or scales.numel()!=kept or ids.unique().numel()!=kept:
            raise ValueError('invalid or duplicate sparse block indices')
        if bool((ids<0).any() or (ids>=total).any()):
            raise ValueError('sparse block index out of range')
        if not bool(torch.isfinite(scales).all()) or bool((scales<0).any()):
            raise ValueError('invalid sparse scales')
        packed=state[KEYS[bits]].detach().cpu().reshape(-1)
        expected=kept*block*bits//8
        if packed.numel()!=expected or packed.dtype!=(torch.int8 if bits==8 else torch.uint8):
            raise ValueError('invalid packed sparse payload')
        if bits==8:
            values=packed.float()
        elif bits==4:
            values=torch.empty(kept*block,dtype=torch.float32)
            values[0::2]=(packed&15).float()-8
            values[1::2]=(packed>>4).float()-8
        else:
            values=torch.empty(kept*block,dtype=torch.float32)
            for i in range(4):values[i::4]=((packed>>(2*i))&3).float()
            zeros=state[prefix+'zero_points'].detach().cpu().reshape(-1)
            if zeros.dtype!=torch.uint8 or zeros.numel()!=kept or bool((zeros>3).any()):
                raise ValueError('invalid INT2 zero points')
            values=values.reshape(kept,block)-zeros.float()[:,None]
        values=values.reshape(kept,block)*scales[:,None]
        result=torch.zeros((total,block),dtype=torch.float32)
        result.index_copy_(0,ids.long(),values)
        lo=state['xyz_bound_min'].detach().cpu().float()
        hi=state['xyz_bound_max'].detach().cpu().float()
        if lo.shape!=(3,) or hi.shape!=(3,) or not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all() and (hi>lo).all()):
            raise ValueError('invalid sparse NTC bounds')
        return {'model.params':result.reshape(shape),'xyz_bound_min':lo,'xyz_bound_max':hi}
    except (KeyError,AttributeError,TypeError,RuntimeError,OverflowError) as exc:
        raise ValueError('malformed sparse NTC checkpoint') from exc
