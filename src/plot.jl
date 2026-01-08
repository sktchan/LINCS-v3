using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using CairoMakie, Statistics, StatsBase, DataFrames, CUDA, Flux

function plot_exp(train_losses, test_losses, model, X_test_masked, y_test_masked, X_train, X_test, 
                  batch_size, n_epochs, save_dir)
    
    fig_loss = Figure(size = (800, 600))
    ax_loss = Axis(fig_loss[1, 1], 
        xlabel="epoch", 
        ylabel="loss (mse)", 
        title="train vs. test loss"
    )
    lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
    lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
    axislegend(ax_loss, position=:rt)
    save(joinpath(save_dir, "loss.png"), fig_loss)

    all_preds = Float32[]
    all_trues = Float32[]
    all_gene_indices = Int[]
    all_column_indices = Int[]

    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
        _, preds_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

        y_cpu = cpu(y_gpu)
        masked_indices = findall(!isnan, y_cpu)
        batch_gene_indices = [idx[1] for idx in masked_indices]
        append!(all_gene_indices, batch_gene_indices)

        append!(all_preds, cpu(preds_masked))
        append!(all_trues, cpu(y_masked))

        batch_col_indices = start_idx:end_idx
        pred_col_indices = [batch_col_indices[idx[2]] for idx in masked_indices]
        append!(all_column_indices, pred_col_indices)
    end

    correlation = cor(all_trues, all_preds)

    # boxplot and histogram combined
    min_val = minimum(all_trues)
    max_val = maximum(all_trues)

    # define bins
    bin_edges = min_val:1.0:max_val
    bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

    stats = Dict()
    x_outliers = Float32[]
    y_outliers = Float32[]
    grouped_preds = Float32[]
    grouped_trues_midpts = Float64[]

    for (i, midpt) in enumerate(bin_midpts) # for each bin
        ind = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
        if !isempty(ind)
            bin_preds = all_preds[ind]
            append!(grouped_preds, bin_preds)
            append!(grouped_trues_midpts, fill(midpt, length(bin_preds)))

            q10 = quantile(bin_preds, 0.10)
            q25 = quantile(bin_preds, 0.25)
            q50 = quantile(bin_preds, 0.50)
            q75 = quantile(bin_preds, 0.75)
            q90 = quantile(bin_preds, 0.90)

            stats[midpt] = (q10=q10, q25=q25, q50=q50, q75=q75, q90=q90)
            outlier_ind = findall(y -> y < q10 || y > q90, bin_preds)
            append!(x_outliers, fill(midpt, length(outlier_ind)))
            append!(y_outliers, bin_preds[outlier_ind])
        end
    end

    midpts_plot = collect(keys(stats))
    q10s = [s.q10 for s in values(stats)]
    q25s = [s.q25 for s in values(stats)]
    q50s = [s.q50 for s in values(stats)]
    q75s = [s.q75 for s in values(stats)]
    q90s = [s.q90 for s in values(stats)] 

    begin
        # setup
        fig_boxhist = Figure(size = (800, 800))
        ax_box = Axis(fig_boxhist[1, 1],
            xlabel="",
            ylabel="predicted expression value",
            title="predicted vs. true expression values"
        )
        ax_hist = Axis(fig_boxhist[2, 1],
            xlabel="true expression value",
            ylabel="count",
            title="distribution of true expression values",
            xticks = 0:5:15,
        )
        linkxaxes!(ax_box, ax_hist)

        # boxplot
        scatter!(ax_box, x_outliers, y_outliers, markersize = 5, alpha = 0.5)
        rangebars!(ax_box, midpts_plot, q10s, q25s, color = :black, whiskerwidth = 0.5)
        rangebars!(ax_box, midpts_plot, q75s, q90s, color = :black, whiskerwidth = 0.5)
        boxplot!(ax_box, grouped_trues_midpts, grouped_preds, range = false, whiskerlinewidth = 0, show_outliers = false)

        # histogram
        hist!(ax_hist, all_trues, bins = bin_edges, strokecolor = :black, strokewidth = 1)
        rowgap!(fig_boxhist.layout, 1, 10)
        display(fig_boxhist)
        save(joinpath(save_dir, "box_hist.png"), fig_boxhist)
    end

    # hexbin plot
    fig_hex = Figure(size = (800, 600))
    ax_hex = Axis(fig_hex[1, 1],
        xlabel="true expression val",
        ylabel="predicted expression val",
        title="predicted vs. true expression density",
        aspect=DataAspect() 
    )
    hexplot = hexbin!(ax_hex, all_trues, all_preds)
    Colorbar(fig_hex[1, 2], hexplot, label="point count")
    save(joinpath(save_dir, "hexbin.png"), fig_hex)

    # error by gene analysis
    absolute_errors = abs.(all_trues .- all_preds)
    df_gene_errors = DataFrame(gene_index = all_gene_indices, absolute_error = absolute_errors)

    # for all pred errors
    begin
        fig_gene_error_scatter = Figure(size = (800, 600))
        ax_gene_error_scatter = Axis(fig_gene_error_scatter[1, 1], title = "prediction error by gene", xlabel = "gene index", ylabel = "absolute prediction error")
        scatter!(ax_gene_error_scatter, all_gene_indices, absolute_errors, alpha=0.5)
        display(fig_gene_error_scatter)
        save(joinpath(save_dir, "gene_vs_error_scatter.png"), fig_gene_error_scatter)
    end

    # for mean pred error
    df_mean_error = combine(groupby(df_gene_errors, :gene_index), :absolute_error => mean => :mean_absolute_error)
    begin
        fig_gene_meanerror = Figure(size = (800, 600))
        ax_gene_meanerror= Axis(fig_gene_meanerror[1, 1], title = "mean absolute error by gene", xlabel = "gene index", ylabel = "mean error")
        scatter!(ax_gene_meanerror, df_mean_error.gene_index, df_mean_error.mean_absolute_error, alpha=0.5)
        display(fig_gene_meanerror)
        save(joinpath(save_dir, "gene_vs_meanerror.png"), fig_gene_meanerror)
    end

    # checking if predicting average
    gene_averages_train = vec(mean(X_train, dims=2)) |> cpu
    masked_indices = findall(!isnan, y_test_masked)
    gene_indices_for_masked_values = getindex.(masked_indices, 1)
    baseline_preds = gene_averages_train[gene_indices_for_masked_values]
    mse_model = mean((all_trues .- all_preds).^2)
    mse_baseline = mean((all_trues .- baseline_preds).^2)

    fig_baseline_hex = Figure(size = (800, 600))
    ax_baseline_hex = Axis(fig_baseline_hex[1, 1], xlabel="true expression val", ylabel="gene average val", title="predicting the average vs. true expression density", aspect=DataAspect())
    hexplot_baseline = hexbin!(ax_baseline_hex, all_trues, baseline_preds)
    Colorbar(fig_baseline_hex[1, 2], hexplot_baseline, label="point count")
    save(joinpath(save_dir, "avg_hexbin.png"), fig_baseline_hex)

    # convert back into ranks for evaluation
    function convert_to_rank(values, ref)
        combined = vcat(values, ref)
        p = sortperm(combined, rev=true)
        ranks = invperm(p)
        return ranks[1]
    end

    reference_matrix = X_test 
    ranked_preds = similar(all_preds, Int)
    ranked_trues = similar(all_trues, Int)

    for i in 1:length(all_preds)
        pred = all_preds[i]
        true_val = all_trues[i]
        col_idx = all_column_indices[i]
        reference_col = reference_matrix[:, col_idx]
        ranked_preds[i] = convert_to_rank(pred, reference_col)
        ranked_trues[i] = convert_to_rank(true_val, reference_col)
    end

    # heatmap
    bin_edges = 1:979 
    h = fit(Histogram, (ranked_trues, ranked_preds), (bin_edges, bin_edges))

    fig_hm = Figure(size = (800, 700))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "true rank",
        ylabel = "predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    Colorbar(fig_hm[1, 2], hm, label = "count (log10)")
    save(joinpath(save_dir, "heatmap.png"), fig_hm)

    return (all_preds=all_preds, all_trues=all_trues, all_gene_indices=all_gene_indices, 
           all_column_indices=all_column_indices, correlation=correlation, 
           mse_model=mse_model, mse_baseline=mse_baseline)
end

# plotting functions for rank (rank_tf) workflow
function plot_rank(train_losses, test_losses, test_rank_errors, model, X_test_masked, y_test_masked, 
                  X, batch_size, n_epochs, n_classes, save_dir)
    
    # loss plot
    fig_loss = Figure(size = (800, 600))
    ax_loss = Axis(fig_loss[1, 1], xlabel="epoch", ylabel="loss (logit-ce)", title="train vs. test loss")
    lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
    lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
    axislegend(ax_loss, position=:rt)
    save(joinpath(save_dir, "loss.png"), fig_loss)

    # rank error plot
    fig_err = Figure(size = (800, 600))
    ax_err = Axis(fig_err[1, 1], xlabel="epoch", ylabel="error", title="mean rank errors")
    lines!(ax_err, 1:n_epochs, test_rank_errors, label="test error", linewidth=2)
    save(joinpath(save_dir, "error.png"), fig_err)

    # collect predictions and compute rank errors
    all_preds = Int[]
    all_trues = Int[]
    all_original_ranks = Int[]
    all_prediction_errors = Int[]

    Flux.testmode!(model)
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

        if isempty(y_masked) continue end

        logits_cpu = cpu(logits_masked)
        y_cpu = cpu(y_masked)
        
        y_cpu_batch = cpu(y_gpu)
        masked_indices_cartesian = findall(y_cpu_batch .!= -100)
        original_ranks_in_batch = [idx[1] for idx in masked_indices_cartesian]

        for i in 1:length(y_cpu)
            true_gene_id = y_cpu[i]
            prediction_logits = logits_cpu[:, i]
            ranked_gene_ids = sortperm(prediction_logits, rev=true)
            predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids)
            
            if !isnothing(predicted_rank)
                error = predicted_rank - 1
                push!(all_prediction_errors, error)
                
                original_rank = original_ranks_in_batch[i] - 1
                push!(all_original_ranks, original_rank)
            end
        end

        predicted_ids = Flux.onecold(logits_masked)
        append!(all_preds, cpu(predicted_ids))
        append!(all_trues, y_cpu)
    end

    # Boxplots
    bin_size = 50
    bin_edges = collect(1:bin_size:n_classes)
    if bin_edges[end] < n_classes
        push!(bin_edges, n_classes + 1)
    end
    bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    grouped_preds = Int[]
    grouped_trues_midpts = Float64[]
    for i in 1:length(bin_edges)-1
        indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
        if !isempty(indices)
            preds_in_bin = all_preds[indices]
            midpoint = bin_midpts[i]
            append!(grouped_preds, preds_in_bin)
            append!(grouped_trues_midpts, fill(midpoint, length(preds_in_bin)))
        end
    end
    fig_box = Figure(size = (800, 600))
    ax_box = Axis(fig_box[1, 1], xlabel="true gene id", ylabel="predicted gene id", title="predicted vs true gene ids")
    boxplot!(ax_box, grouped_trues_midpts, grouped_preds, width=bin_size*0.5)
    save(joinpath(save_dir, "boxplot.png"), fig_box)

    # Hexbin
    fig_hex = Figure(size = (800, 700))
    ax_hex = Axis(fig_hex[1, 1], xlabel="true gene id", ylabel="predicted gene id", title="predicted vs true gene id density", aspect=DataAspect())
    hexplot = hexbin!(ax_hex, all_trues, all_preds)
    Colorbar(fig_hex[1, 2], hexplot, label="point count")
    save(joinpath(save_dir, "hexbin.png"), fig_hex)

    # MASKED PREDICTION ERROR BY RANK (SCATTER)
    fig_rank_error_scatter = Figure(size = (800, 600))
    ax_rank_error_scatter = Axis(fig_rank_error_scatter[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "prediction error", title = "prediction error by rank")
    scatter!(ax_rank_error_scatter, all_original_ranks, all_prediction_errors, markersize=4, alpha=0.3)
    save(joinpath(save_dir, "rank_vs_error_scatter.png"), fig_rank_error_scatter)

    # MEAN MASKED PREDICTION ERROR BY RANK (SCATTER)
    df_rank_errors = DataFrame(original_rank = all_original_ranks, prediction_error = all_prediction_errors)
    avg_errors = combine(groupby(df_rank_errors, :original_rank), :prediction_error => mean => :avg_error)
    fig_rank_error_line = Figure(size = (800, 600))
    ax_rank_error_line = Axis(fig_rank_error_line[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "mean prediction error", title = "mean prediction error by rank")
    scatter!(ax_rank_error_line, avg_errors.original_rank, avg_errors.avg_error, alpha = 0.5)
    save(joinpath(save_dir, "rank_vs_avgerror_scatter.png"), fig_rank_error_line)

    return (all_preds=all_preds, all_trues=all_trues, all_original_ranks=all_original_ranks, 
           all_prediction_errors=all_prediction_errors, avg_errors=avg_errors)
end

