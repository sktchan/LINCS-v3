#=
nsys profile -o output/report julia example.jl 
nsys stats output/report.nsys-rep
nsys analyze output/report.nsys-rep
=#

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using Dates, JLD2
using LincsProject
using Flux, ProgressBars, CUDA, Statistics
using NVTX, Zygote
using BenchmarkTools

Zygote.@adjoint function NVTX.range_push(args...; kwargs...)
    y = NVTX.range_push(args...; kwargs...)
    return y, Δ -> nothing
end

Zygote.@adjoint function NVTX.range_pop()
    NVTX.range_pop()
    return nothing, Δ -> nothing
end

include("../src/params.jl")
include("../src/fxns.jl")

CUDA.device!(0)

start_time = now()
NVTX.range_push(message="data load")
data = load(data_path)["filtered_data"]
NVTX.range_pop()

NVTX.range_push(message="data rank")
gene_medians = vec(median(data.expr, dims=2)) .+ 1e-10
X = rank_genes(data.expr, gene_medians)
X = X[:, 1:1000] # subset for profiling easier :)
NVTX.range_pop()

n_features = size(X, 1) + 2
n_classes = size(X, 1)
n_genes = size(X, 1)
MASK_ID = (n_classes + 1)
CLS_ID = n_genes + 2
CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
X = vcat(CLS_VECTOR, X)

NVTX.range_push(message="split/mask")
X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, -100, MASK_ID)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, -100, MASK_ID)
NVTX.range_pop()

struct PosEnc{A<:AbstractArray}
    pe_matrix::A
end

function PosEnc(embed_dim::Int, max_len::Int) # max_len is usually maximum length of sequence but here it is just len(genes)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
    return PosEnc(cu(pe_matrix)) # removed cu(pe_matrix)
end

Flux.@functor PosEnc

function (pe::PosEnc)(input::Float32Matrix3DType)
    seq_len = size(input,2)
    return input .+ @view(pe.pe_matrix[:, 1:seq_len]) # adds positional encoding to input embeddings
end

### transformer
struct Transf{A,D,N,M}
    mha::A
    att_dropout::D
    att_norm::N
    mlp::M 
    mlp_norm::N
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix3DType) # input shape: embed_dim × seq_len × batch_size
    NVTX.range_push(message="tf-attn")
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    NVTX.range_pop()

    NVTX.range_push(message="tf-drop/norm")
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    NVTX.range_pop()

    NVTX.range_push(message="tf-reshape fwd")
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    NVTX.range_pop()

    NVTX.range_push(message="tf-mlp")
    mlp_out = tf.mlp(reshaped)
    NVTX.range_pop()

    NVTX.range_push(message="tf-reshape bkw/resid")
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped
    NVTX.range_pop()
    return tf_output
end

### full model as << ranked data --> token embedding --> position embedding --> transformer --> classifier head >>
struct Model{E, P, D, T, C}
    embedding::E
    pos_encoder::P
    pos_dropout::D
    transformer::T
    classifier::C
end

function Model(;
    input_size::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    embedding = Flux.Embedding(input_size => embed_dim)
    pos_encoder = PosEnc(embed_dim, input_size)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )
    return Model(embedding, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::IntMatrix2DType)
    NVTX.range_push(message="m-embedding")
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    NVTX.range_pop()

    NVTX.range_push(message="m-transformer")
    transformed = model.transformer(encoded_dropped)
    NVTX.range_pop()

    NVTX.range_push(message="m-classifier")
    logits_output = model.classifier(transformed)
    NVTX.range_pop()
    
    return logits_output
end

### training ###

model = Model(
    input_size=n_features,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#=
loss: cross-entropy between the model’s predicted distribution and the true token at each masked position
compute the loss by iterating over masked positions OR by using a mask in the loss function
=#
function loss(model::Model, x, y, mode::String)
    NVTX.range_push(message="loss-fwd")
    logits = model(x)  # (n_classes, seq_len, batch_size)
    NVTX.range_pop()

    NVTX.range_push(message="loss-reshape")
    logits_flat = reshape(logits, size(logits, 1), :) # (n_classes, seq_len*batch_size)
    NVTX.range_pop()

    NVTX.range_push(message="loss-find masks")
    y_flat = vec(y) # (seq_len*batch_size) column vec
    mask = y_flat .!= -100 # bit vec, where sum = n_masked
    logits_masked = logits_flat[:, mask] # (n_classes, n_masked)
    y_masked = y_flat[mask] # (n_masked) column vec
    NVTX.range_pop()

    NVTX.range_push(message="loss-onehotbatch")
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes) # (n_classes, n_masked)
    NVTX.range_pop()

    NVTX.range_push(message="loss-crossentropy")
    if mode == "train"
        l = Flux.logitcrossentropy(logits_masked, y_oh) 
        NVTX.range_pop()
        return l
    end
    if mode == "test"
        l = Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
        NVTX.range_pop()
        return l
    end
end

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]

all_preds = Int[]
all_trues = Int[]
all_original_ranks = Int[]
all_prediction_errors = Int[]

# CUDA.pin(X_train_masked)
# CUDA.pin(y_train_masked)
# CUDA.pin(X_test_masked)
# CUDA.pin(y_test_masked)

function train_model(model, opt, X_train_masked, y_train_masked, X_test_masked, y_test_masked, n_epochs)
    for epoch in ProgressBar(1:n_epochs)
        epoch_losses = Float32[]

        NVTX.range_push(message="train")
        for start_idx in 1:batch_size:size(X_train_masked, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))

            NVTX.range_push(message="cpu to gpu")
            x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
            y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
            NVTX.range_pop()
            
            NVTX.range_push(message="gradient calc")
            loss_val, grads = Flux.withgradient(model) do m
                loss(m, x_gpu, y_gpu, "train")
            end
            NVTX.range_pop()

            NVTX.range_push(message="update")
            Flux.update!(opt, model, grads[1])
            NVTX.range_pop()

            NVTX.range_push(message="recalc loss")
            loss_val = loss(model, x_gpu, y_gpu, "train")
            NVTX.range_pop()
            push!(epoch_losses, loss_val)
        end
        push!(train_losses, mean(epoch_losses))
        NVTX.range_pop()

        test_epoch_losses = Float32[]
        epoch_rank_errors = Int[]

        NVTX.range_push(message="test")
        for start_idx in 1:batch_size:size(X_test_masked, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))

            NVTX.range_push(message="cpu to gpu")
            x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
            y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
            NVTX.range_pop()

            NVTX.range_push(message="loss calc")
            test_loss_val, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")
            push!(test_epoch_losses, test_loss_val)
            NVTX.range_pop()

            if isempty(y_masked) continue end

            NVTX.range_push(message="gpu to cpu")
            logits_cpu = cpu(logits_masked)
            y_cpu = cpu(y_masked)
            NVTX.range_pop()
            
            if epoch == n_epochs
                NVTX.range_push(message="find indices")
                y_cpu_batch = cpu(y_gpu)
                masked_indices_cartesian = findall(y_cpu_batch .!= -100)
                original_ranks_in_batch = [idx[1] for idx in masked_indices_cartesian]
                NVTX.range_pop()
            end

            NVTX.range_push(message="get ranks/errors")
            for i in 1:length(y_cpu)
                true_gene_id = y_cpu[i]
                prediction_logits = logits_cpu[:, i]
                ranked_gene_ids = sortperm(prediction_logits, rev=true)
                predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids)
                
                if !isnothing(predicted_rank)
                    error = predicted_rank - 1
                    push!(epoch_rank_errors, error)
                    
                    if epoch == n_epochs
                        original_rank = original_ranks_in_batch[i] - 1
                        push!(all_original_ranks, original_rank)
                        push!(all_prediction_errors, error)
                    end
                end
            end
            NVTX.range_pop()
            
            NVTX.range_push(message="onecold")
            if epoch == n_epochs
                predicted_ids = Flux.onecold(logits_masked)
                append!(all_preds, cpu(predicted_ids))
                append!(all_trues, y_cpu)
            end
            NVTX.range_pop()
        end
        push!(test_losses, mean(test_epoch_losses))
        NVTX.range_pop()

        if !isempty(epoch_rank_errors)
            push!(test_rank_errors, mean(epoch_rank_errors))
        else
            push!(test_rank_errors, NaN32)
        end
    end
end

@btime train_model(model, opt, X_train_masked, y_train_masked, X_test_masked, y_test_masked, n_epochs)