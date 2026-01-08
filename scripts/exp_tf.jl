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

Model = ExpModel

CUDA.device!(0)

start_time = now()
data = load(data_path)["filtered_data"]
X = data.expr

n_genes = size(X, 1)
n_classes = 1 

X_train, X_test, test_indices, train_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_raw(X_train)
X_test_masked, y_test_masked = mask_raw(X_test)

model = Model(
    seq_len=n_genes,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes, # n_classes is 1
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

train_losses, test_losses = train(model, opt, n_epochs, batch_size, X_train_masked, y_train_masked, X_test_masked, y_test_masked)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "exp_tf", timestamp)
mkpath(save_dir)

plot_results = plot_exp(train_losses, test_losses, model, X_test_masked, y_test_masked, X_train, X_test, 
                        batch_size, n_epochs, save_dir)

save_exp(model, train_losses, test_losses, X_test_masked, y_test_masked, X_test, 
         test_indices, train_indices, plot_results, start_time, save_dir, n_epochs)
