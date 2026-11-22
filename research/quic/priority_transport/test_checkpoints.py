"""Sparse weights must reconstruct exact positions and reject malformed metadata."""
import importlib
import pytest
import torch


def fixture(bits):
    prefix=f'model.params_sparse_int{bits}_'
    state={'xyz_bound_min':torch.zeros(3),'xyz_bound_max':torch.ones(3)}
    state.update({prefix+'block_indices':torch.tensor([1],dtype=torch.int32),
                  prefix+'scales':torch.tensor([.5]),prefix+'block_size':torch.tensor(64),
                  prefix+'total_blocks':torch.tensor(2),prefix+'keep_blocks':torch.tensor(1),
                  prefix+'shape':torch.tensor([128])})
    if bits==8:
        state['model.params_sparse_qint8']=torch.tensor([1,-2]*32,dtype=torch.int8)
        state[prefix+'quantization']='symmetric_sparse_blockwise_int8'
        expected=torch.tensor([.5,-1.]*32)
    elif bits==4:
        state['model.params_sparse_qint4_packed']=torch.full((32,),0x69,dtype=torch.uint8)
        state[prefix+'quantization']='offset_signed_int4_packed'
        expected=torch.tensor([.5,-1.]*32)
    else:
        state['model.params_sparse_qint2_packed']=torch.full((16,),0xe4,dtype=torch.uint8)
        state[prefix+'zero_points']=torch.tensor([2],dtype=torch.uint8)
        state[prefix+'quantization']='block_affine_int2_u0123_v1'
        expected=torch.tensor([-1.,-.5,0.,.5]*16)
    return state,torch.cat([torch.zeros(64),expected])


def decoder():
    return importlib.import_module('research.quic.priority_transport.checkpoints').decode_sparse_checkpoint


@pytest.mark.parametrize('bits',[2,4,8])
def test_sparse_decode_positions_and_packing(bits):
    state,expected=fixture(bits)
    actual=decoder()(state)
    assert set(actual)=={'model.params','xyz_bound_min','xyz_bound_max'}
    assert actual['model.params'].dtype==torch.float32
    assert torch.equal(actual['model.params'],expected)
    assert 'model.params' not in state


@pytest.mark.parametrize('kind',['duplicate','range','scale','length','format','shape'])
def test_reject_malformed_sparse_state(kind):
    state,_=fixture(4)
    prefix='model.params_sparse_int4_'
    if kind=='duplicate':state[prefix+'block_indices']=torch.tensor([1,1])
    if kind=='range':state[prefix+'block_indices']=torch.tensor([2])
    if kind=='scale':state[prefix+'scales']=torch.tensor([float('nan')])
    if kind=='length':state['model.params_sparse_qint4_packed']=torch.zeros(31,dtype=torch.uint8)
    if kind=='format':state[prefix+'quantization']='unknown'
    if kind=='shape':state[prefix+'shape']=torch.tensor([0])
    with pytest.raises(ValueError):decoder()(state)


def test_dense_state_passthrough():
    state={'model.params':torch.ones(128)}
    assert decoder()(state) is state


def test_native_priority_provider_loads_sparse_checkpoint(tmp_path):
    from live_priority import LivePriorityState
    state,expected=fixture(4)
    path=tmp_path/'NTC_000000.pth'
    torch.save(state,path)
    provider=object.__new__(LivePriorityState)
    assert torch.equal(provider._load_state_dict_fast(str(path))['model.params'],expected)
