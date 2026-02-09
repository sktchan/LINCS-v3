using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using DataFrames, Dates, StatsBase, JLD2
using LincsProject
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MultivariateStats

include("../src/params.jl")
include("../src/fxns.jl")
include("../src/plot.jl")
include("../src/save.jl")

CUDA.device!(0)

data = load(data_path)["filtered_data"]
raw_data = data.expr

max_cpts = 100
init_pca = fit(PCA, raw_data; maxoutdim=max_cpts)

vars = principalvars(init_pca)
totalvar = tprincipalvar(init_pca) + tresidualvar(init_pca)
ratio = vars ./ totalvar
cum_ratio = cumsum(ratio)

begin
    fig1 = Figure(size=(600,400))
    ax1 = Axis(fig1[1,1])
    barplot!(ax1, 1:length(cum_ratio), cum_ratio)
    display(fig1)
    save("/home/golem/scratch/chans/lincsv3/plots/trt/pca/cumvar.png", fig1)
end

begin
    fig2 = Figure(size=(600,400))
    ax2 = Axis(fig2[1,1])
    barplot!(ax2, 1:length(ratio), ratio)
    display(fig2)
    save("/home/golem/scratch/chans/lincsv3/plots/trt/pca/scree.png", fig2)
end

proj = predict(init_pca, raw_data)

begin
    fig3 = Figure(size=(600,400))
    ax3 = Axis(fig3[1,1])
    scatter!(ax3, proj[1,:], proj[2,:], markersize=2)
    display(fig3)
    save("/home/golem/scratch/chans/lincsv3/plots/trt/pca/pcs.png", fig3)
end

# i guess we can use ~ 50 cpts?

struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm # this is the normalization aspect
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
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


struct Model
    embedding::Flux.Embedding
    # pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
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
    return Model(embedding, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::IntMatrix2DType)
    embedded = model.embedding(input)
    # encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    logits_output = model.classifier(transformed)
    return logits_output
end