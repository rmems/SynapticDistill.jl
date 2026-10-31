# SPDX-License-Identifier: MIT OR Apache-2.0

"""
SynapticDistill.jl

Modular online training for spiking neural networks in Julia — E-prop, OTTT, and more.
Works with pure SNNs or hybrid teacher-student systems.
"""
module SynapticDistill

using LinearAlgebra
using Statistics
using MLUtils
using Zygote

# Core data structures
include("types.jl")
export SpikeBatch, TraceBatch, TrainingState

# Utilities
include("utils.jl")

# Training rules
include("rules/surrogate.jl")
include("rules/eprop.jl")
include("rules/ottt.jl")
export surrogate_heaviside, surrogate_sigmoid, surrogate_exponential
export update_eprop!, update_ottt!

# Q8.8 FPGA `.mem` export (silicon-bridge compatible)
include("export_q88.jl")
export to_q88, q88_to_float, q88_hex, read_q88_mem, export_q88_mem

# Main training loop
include("training.jl")
export ModelStep, train_step!

end # module
