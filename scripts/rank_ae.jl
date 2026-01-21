using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, Dates, StatsBase, JLD2, MLUtils
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

include("../src/params.jl")
include("../src/fxns.jl")
include("../src/plot.jl")
include("../src/save.jl")

CUDA.device!(0)

start_time = now()
data = load(data_path)["filtered_data"]

gene_medians = vec(median(data.expr, dims=2)) .+ 1e-10
X = Int32.(rank_genes(data.expr, gene_medians))
# X = rank_genes(data.expr, gene_medians)

n_features = size(X, 1) + 2
n_classes = size(X, 1)
n_genes = size(X, 1)
MASK_ID = (n_classes + 1)
CLS_ID = n_genes + 2
CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
X = Int32.(vcat(CLS_VECTOR, X))

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)
X_train_masked, y_train_masked = mask_input(X_train, mask_ratio, -100, MASK_ID)
X_test_masked, y_test_masked = mask_input(X_test, mask_ratio, -100, MASK_ID)

VOCAB_SIZE = n_genes + 1 

### model ###


### training ###


### plot/save ###

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", dataset, "rank_ae", timestamp)
mkpath(save_dir)

plot_loss(n_epochs, train_losses, test_losses, save_dir, "logit-ce")
plot_rank_error(n_epochs, test_rank_errors, save_dir)
plot_boxplot(n_classes, all_trues, all_preds, save_dir)
plot_hexbin(all_trues, all_preds, "gene id", save_dir)
plot_prediction_error(all_original_ranks, all_prediction_errors, save_dir)

avg_errors = plot_mean_prediction_error(all_original_ranks, all_prediction_errors, save_dir)
cs, cp = plot_ranked_heatmap(all_trues, all_preds, save_dir, true)


log_model(model, save_dir)
# embeddings = get_profile_embeddings(X, model)

# log run info
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

log_info(train_indices, test_indices, embeddings, n_epochs, 
                    train_losses, test_losses, all_preds, all_trues, 
                    all_original_ranks, all_prediction_errors, 
                    avg_errors, X_test_masked, y_test_masked, X_test)


log_params(gpu_info, dataset, mask_ratio, batch_size, n_epochs, 
                    embed_dim, hidden_dim, n_heads, n_layers, lr, drop_prob, 
                    additional_notes, run_hours, run_minutes)