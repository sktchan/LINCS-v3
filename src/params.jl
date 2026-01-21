# dataset
# data_path = "data/lincs_trt_untrt_data.jld2"
# dataset = "trt"
data_path = "data/lincs_untrt_data.jld2"
dataset = "untrt"

# params
batch_size = 40
n_epochs = 1
embed_dim = 128
hidden_dim = 256
n_heads = 2
n_layers = 4
drop_prob = 0.05
lr = 0.001
mask_ratio = 0.1

# ae
# latent_1 = 734
# latent_2 = 490
# latent_3 = 246

latent_1 = 400
latent_2 = 300
latent_3 = 200

# notes
gpu_info = "oni"
additional_notes = "test run for new script"

# matrix types
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}
