using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using Dates, JLD2
using LincsProject
using Flux, ProgressBars, CUDA, Statistics, BenchmarkTools

include("../src/params.jl")
include("../src/fxns.jl")

CUDA.device!(0)

start_time = now()
data = load(data_path)["filtered_data"]

gene_medians = vec(median(data.expr, dims=2)) .+ 1e-10
X = rank_genes(data.expr, gene_medians)
X = X[:, 1:1000] # subset for profiling easier :)

n_features = size(X, 1) + 2
n_classes = size(X, 1)
n_genes = size(X, 1)
MASK_ID = (n_classes + 1)
CLS_ID = n_genes + 2
CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
X = vcat(CLS_VECTOR, X)

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, -100, MASK_ID)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, -100, MASK_ID)

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
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

# for faster?
function fast_gelu(x)
    return 0.5f0 * x * (1.0f0 + tanh(0.79788456f0 * (x + 0.044715f0 * x^3)))
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
        Flux.Dense(embed_dim => embed_dim, fast_gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )
    return Model(embedding, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::IntMatrix2DType)
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    logits_output = model.classifier(transformed)
    
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
    logits = model(x)  # (n_classes, seq_len, batch_size)
    logits_flat = reshape(logits, size(logits, 1), :) # (n_classes, seq_len*batch_size)
    y_flat = vec(y) # (seq_len*batch_size) column vec
    mask = y_flat .!= -100 # bit vec, where sum = n_masked
    logits_masked = logits_flat[:, mask] # (n_classes, n_masked)
    y_masked = y_flat[mask] # (n_masked) column vec
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes) # (n_classes, n_masked)

    if mode == "train"
        return Flux.logitcrossentropy(logits_masked, y_oh) 
    end
    if mode == "test"
        return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]

all_preds = Int[]
all_trues = Int[]
all_original_ranks = Int[]
all_prediction_errors = Int[]

function train_model(model, opt, X_train_masked, y_train_masked, X_test_masked, y_test_masked, n_epochs)
    for epoch in ProgressBar(1:n_epochs)
        epoch_losses = Float32[]

        for start_idx in 1:batch_size:size(X_train_masked, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))
            x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
            y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
            
            loss_val, grads = Flux.withgradient(model) do m
                loss(m, x_gpu, y_gpu, "train")
            end
            Flux.update!(opt, model, grads[1])
            loss_val = loss(model, x_gpu, y_gpu, "train")
            push!(epoch_losses, loss_val)
        end
        push!(train_losses, mean(epoch_losses))

        test_epoch_losses = Float32[]
        epoch_rank_errors = Int[]
        
        for start_idx in 1:batch_size:size(X_test_masked, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
            x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
            y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

            test_loss_val, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")
            push!(test_epoch_losses, test_loss_val)

            if isempty(y_masked) continue end

            logits_cpu = cpu(logits_masked)
            y_cpu = cpu(y_masked)
            
            if epoch == n_epochs
                y_cpu_batch = cpu(y_gpu)
                masked_indices_cartesian = findall(y_cpu_batch .!= -100)
                original_ranks_in_batch = [idx[1] for idx in masked_indices_cartesian]
            end

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

            if epoch == n_epochs
                predicted_ids = Flux.onecold(logits_masked)
                append!(all_preds, cpu(predicted_ids))
                append!(all_trues, y_cpu)
            end
        end
        push!(test_losses, mean(test_epoch_losses))
        if !isempty(epoch_rank_errors)
            push!(test_rank_errors, mean(epoch_rank_errors))
        else
            push!(test_rank_errors, NaN32)
        end
    end
end

@btime train_model(model, opt, X_train_masked, y_train_masked, X_test_masked, y_test_masked, n_epochs)

#=
btime results!

regular: 14.421 s (1559781 allocations: 3.63GiB)

type-ing posenc: 15.443 s (1557427 allocations: 3,63GiB)
type-ing posenc + tf: 15.422 s (1545485 allocations: 3.62GiB)
type-ing posenc + tf + model: 15.515 s (1542339 allocations: 3.61GiB) *** using this

@view in posenc instead of index: 15.484 s (1547401 allocations: 3.63 GiB) 
fast_gelu: 15.567 s (1540503 allocations: 3.62 GiB)
=#