using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, Dates, StatsBase, JLD2
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

include("../src/params.jl")
include("../src/struct.jl")
include("../src/fxns.jl")
include("../src/train.jl")
include("../src/plot.jl")
include("../src/save.jl")

Model = RankModel

CUDA.device!(0)

start_time = now()
data = load(data_path)["filtered_data"]

gene_medians = vec(median(data.expr, dims=2)) .+ 1e-10
@time X = sort_gene(data.expr, gene_medians) # lookup table of indices from highest rank to lowest rank gene

n_features = size(X, 1) + 2
n_classes = size(X, 1)
n_genes = size(X, 1)
MASK_ID = (n_classes + 1)
CLS_ID = n_genes + 2
CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
X = vcat(CLS_VECTOR, X)

X_train, X_test, test_indices, train_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_rank(X_train)
X_test_masked, y_test_masked = mask_rank(X_test)

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

train_losses, test_losses, test_rank_errors = train(model, opt, n_epochs, batch_size, X_train_masked, y_train_masked, X_test_masked, y_test_masked)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "rank_tf", timestamp)
mkpath(save_dir)

plot_results = plot_rank(train_losses, test_losses, test_rank_errors, model, X_test_masked, y_test_masked, 
                         X, batch_size, n_epochs, n_classes, save_dir)

save_rank(model, train_losses, test_losses, X, X_test_masked, y_test_masked, 
          test_indices, train_indices, plot_results, start_time, save_dir, 
          n_epochs, batch_size)
