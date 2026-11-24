"""Paired weights must preserve values and reject corrupt/mixed deliveries."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from paired_affine import pack_codes,unpack_codes,decode_to_dense_state
from paired_groups import build_groups,assemble


def state(bits):
    return dict(format='qntc_sparse_block_affine_v1',bits=bits,block_size=64,shape=(512,),
                indices=torch.tensor([0,2,5,7],dtype=torch.int32),
                codes=torch.from_numpy(pack_codes(np.full(256,(1<<bits)-1,dtype=np.uint8),bits)),
                scales=torch.tensor([.25,.5,.75,1.],dtype=torch.float32),
                zero_points=torch.zeros(4,dtype=torch.uint8),
                xyz_bound_min=torch.tensor([-1.,-2.,-3.]),xyz_bound_max=torch.tensor([1.,2.,3.]))


def prepared(root,bits):
    groups=build_groups(state(bits),7,np.array([7,0,5,2],dtype=np.int32))
    folder=root/f'int{bits}';folder.mkdir()
    for tier,raw in enumerate(groups):(folder/f'P{tier}.bin').write_bytes(raw)
    record={f'P{i}':dict(payload_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for i,raw in enumerate(groups)}
    return folder,dict(frame=7,retained_blocks=4,variants={str(bits):dict(groups=record)})


@pytest.mark.parametrize('bits',[2,4])
def test_paired_base_and_full_model_are_exact(tmp_path,bits):
    folder,manifest=prepared(tmp_path,bits)
    base,count=assemble(folder,manifest,1)
    assert count==1
    assert torch.count_nonzero(base['model.params'][:448])==0
    assert torch.equal(base['model.params'][448:],torch.full((64,),float((1<<bits)-1)))
    full,count=assemble(folder,manifest,4)
    assert count==4
    assert torch.equal(full['model.params'],decode_to_dense_state(state(bits))['model.params'])


@pytest.mark.parametrize('tiers',[0,5])
def test_invalid_tier_count_fails_clearly(tmp_path,tiers):
    folder,manifest=prepared(tmp_path,2)
    with pytest.raises(ValueError,match='tier|group'):assemble(folder,manifest,tiers)


def test_mixed_precision_is_rejected_even_with_matching_transport_hash(tmp_path):
    folder,manifest=prepared(tmp_path,2)
    other,_=prepared(tmp_path,4)
    body=(other/'P1.bin').read_bytes();(folder/'P1.bin').write_bytes(body)
    manifest['variants']['2']['groups']['P1']=dict(payload_bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError,match='Mixed precision'):assemble(folder,manifest,2)


def test_corrupt_received_group_is_rejected(tmp_path):
    folder,manifest=prepared(tmp_path,4)
    data=bytearray((folder/'P0.bin').read_bytes());data[-1]^=1;(folder/'P0.bin').write_bytes(data)
    with pytest.raises(ValueError,match='Hash mismatch'):assemble(folder,manifest,1)


@pytest.mark.parametrize('bits',[2,4])
def test_padding_cannot_hide_extra_weight_codes(bits):
    packed=pack_codes(np.array([1],dtype=np.uint8),bits)
    assert unpack_codes(packed,1,bits).tolist()==[1]
    packed[-1]|=np.uint8(1<<bits)
    with pytest.raises(ValueError,match='padding'):unpack_codes(packed,1,bits)
