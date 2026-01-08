# dataset
data_path = "data/lincs_trt_untrt_data.jld2"
dataset = "trt"
# data_path = "data/lincs_untrt_data.jld2"
# dataset = "untrt"

# params
batch_size = 210
n_epochs = 1
embed_dim = 128
hidden_dim = 256
n_heads = 2
n_layers = 4
drop_prob = 0.05
lr = 0.001
mask_ratio = 0.1

# notes
gpu_info = "oni"
additional_notes = "test"

# matrix types
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}
MASK_VALUE = -1.0f0
