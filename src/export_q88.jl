# SPDX-License-Identifier: MIT OR Apache-2.0

# Q8.8 `.mem` export, byte-compatible with rmems/silicon-bridge
# (`src/fpga_export.rs`: `encode_q88_signed_full`, `write_mem_file`).
#
# Contract (mirrors silicon-bridge):
#   * word      : signed two's-complement Int16, value * 256
#   * rounding  : truncate toward zero (NOT round-to-nearest)
#   * overflow  : saturate to Int16 (-128.0 -> 8000, >=127.99609375 -> 7FFF)
#   * NaN       : 0
#   * on disk   : one uppercase 4-digit hex word per line, `\n` terminated
#   * weights   : row-major dense, addr = neuron * n_inputs + input

"""
    to_q88(x) -> Int16

Encode `x` as signed Q8.8 exactly like silicon-bridge `encode_q88_signed_full`:
truncate `x * 256` toward zero, saturate at the `Int16` limits, `NaN` -> `0`.
"""
function to_q88(x::Real)
    isnan(x) && return Int16(0)
    # Float32 product matches Rust's `(value * 256.0) as i16` for f32 inputs.
    s = Float32(x) * 256f0
    s >= 32767f0 && return typemax(Int16)
    s <= -32768f0 && return typemin(Int16)
    return Int16(trunc(s))
end

"""
    q88_to_float(w::Int16) -> Float32

Decode a signed Q8.8 word (silicon-bridge `q88_signed_to_f32`).
"""
q88_to_float(w::Int16) = Float32(w) / 256f0

"""
    q88_hex(x) -> String

Four uppercase hex digits of the two's-complement Q8.8 word for `x`
(silicon-bridge `format_q88_hex`).
"""
q88_hex(x::Real) = uppercase(string(reinterpret(UInt16, to_q88(x)), base = 16, pad = 4))

"""
    read_q88_mem(path) -> Vector{Float32}

Parse a `.mem` file (one 4-digit hex word per line) as signed Q8.8.
"""
function read_q88_mem(path::AbstractString)
    out = Float32[]
    for (i, line) in enumerate(eachline(path))
        tok = strip(line)
        isempty(tok) && continue
        length(tok) == 4 || error("$path:$i: Q8.8 word must be 4 hex digits, got $(repr(tok))")
        push!(out, q88_to_float(reinterpret(Int16, parse(UInt16, tok, base = 16))))
    end
    return out
end

function _write_q88_mem(path::AbstractString, values)
    open(path, "w") do io
        for v in values
            write(io, q88_hex(v), '\n')
        end
    end
    return path
end

"""
    export_q88_mem(dir; thresholds, weights, decay, output_weights=nothing) -> Vector{String}

Write a silicon-bridge-compatible Q8.8 `.mem` bundle into `dir`:
`parameters.mem` (thresholds), `parameters_weights.mem` (hidden weights,
row-major `neuron x input`), `parameters_decay.mem`, and, when given,
`parameters_output_weights.mem`.

`weights` is an `N x n_inputs` matrix; `thresholds` and `decay` have length `N`.
`output_weights` (optional) is an `N x K` matrix, flattened row-major.
`Inf` saturates; `NaN` becomes `0`. Returns the written paths.
"""
function export_q88_mem(dir::AbstractString;
                        thresholds::AbstractVector{<:Real},
                        weights::AbstractMatrix{<:Real},
                        decay::AbstractVector{<:Real},
                        output_weights::Union{Nothing, AbstractMatrix{<:Real}} = nothing)
    n = size(weights, 1)
    n > 0 || throw(ArgumentError("weights must have at least one neuron row"))
    length(thresholds) == n ||
        throw(DimensionMismatch("thresholds has $(length(thresholds)) entries, expected $n"))
    length(decay) == n ||
        throw(DimensionMismatch("decay has $(length(decay)) entries, expected $n"))
    if output_weights !== nothing && size(output_weights, 1) != n
        throw(DimensionMismatch("output_weights has $(size(output_weights, 1)) rows, expected $n"))
    end
    mkpath(dir)
    rowmajor(M) = (M[i, j] for i in axes(M, 1) for j in axes(M, 2))
    paths = [
        _write_q88_mem(joinpath(dir, "parameters.mem"), thresholds),
        _write_q88_mem(joinpath(dir, "parameters_weights.mem"), rowmajor(weights)),
        _write_q88_mem(joinpath(dir, "parameters_decay.mem"), decay),
    ]
    output_weights === nothing ||
        push!(paths, _write_q88_mem(joinpath(dir, "parameters_output_weights.mem"),
                                    rowmajor(output_weights)))
    return paths
end
