#!/bin/bash
# NCCL settings for multi-GPU DDP on SCC.
#
# Job 7362778 deadlocked on scc-f04: the log stopped dead at "All DDP processes
# registered", and all four ranks sat at 100% GPU utilisation holding only ~2 GB
# each -- NCCL spin-waiting inside a collective, not computing. scc-f04/f05 are
# PCIe-only A40 nodes where GPU-direct P2P and /dev/shm collectives can hang.
#
# Same conclusion distillation_nerc/scripts_20k/scc_nccl_env.sh reached; this is
# a standalone copy so ldm_official does not reach into that tree, with the
# torch-level names adjusted for torch 1.11 (the TORCH_NCCL_* spellings are
# PyTorch 2.x and are silently ignored here).

# Avoid flaky GPU-direct P2P on PCIe-only topologies.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-LOC}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

# Shared nodes can deadlock on SHM collectives; prefer sockets.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

# torch 1.11 reads the un-prefixed names. Without this a NCCL hang spins
# forever instead of raising -- which is exactly how 92 minutes were lost.
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
# torch 2.x spelling of the same switch (the venv setup_env.sh builds is torch 2.x).
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ "${DEBUG_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
else
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi
