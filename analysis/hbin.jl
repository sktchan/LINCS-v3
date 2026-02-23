# for reformatting heatmaps/hexbins after running model

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using JLD2, CairoMakie, StatsBase

all_trues = load("/home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54/predstrues.jld2")["all_trues"]
all_preds = load("/home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54/predstrues.jld2")["all_preds"]

cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

# exp val

begin
    fig_hex = Figure(size = (800, 700))
    ax_hex = Axis(fig_hex[1, 1],
        # backgroundcolor = to_colormap(:viridis)[1], 
        xlabel="true expression value",
        ylabel="predicted expression value"
        # title="predicted vs. true gene id density"
        # aspect=DataAspect() 
    )
    hexplot = hexbin!(ax_hex, all_trues, all_preds, cellsize = (0.06,0.06), colorscale = log10)
    # text!(ax_hex, 0, 1050, align = (:left, :top), text = "Pearson: $cp")
    Colorbar(fig_hex[1, 2], hexplot, label="point count (log10)")
    display(fig_hex)
end
save_dir = "/home/golem/scratch/chans/lincsv2/plots/untrt/TEST_rank_tf/baseline"
save(joinpath(save_dir, "exp_nn_hbin.png"), fig_hex)
print(cs)



# rank id

# # to sort x axis
sorted_indices_by_mean = load("/home/golem/scratch/chans/lincsv2/plots/untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
gene_id_to_rank_map = invperm(sorted_indices_by_mean);
sorted_trues = gene_id_to_rank_map[all_trues];
sorted_preds = gene_id_to_rank_map[all_preds];

bin_edges = 1:979 
h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))
begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp, digits=4))", color = :white)
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end
save_dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/rank_tf/2026-02-19_22-54"
save(joinpath(save_dir, "heatmap.png"), fig_hm)





# for rank nn

rank_data = load("/home/golem/scratch/chans/lincsv3/plots/untrt/rank_nn/2026-01-14_15-26/rankedpredstrues.jld2")

sorted_trues = rank_data["ranked_trues"]
sorted_preds = rank_data["ranked_preds"]

cp_rank = cor(sorted_trues, sorted_preds)

bin_edges = 1:979 
h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))

begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp_rank, digits=4))", color = :white)
    
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end


save_dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/rank_nn/2026-01-14_15-26"
save(joinpath(save_dir, "heatmap.png"), fig_hm)





# for oh_rank_nn

base_path = "/home/golem/scratch/chans/lincsv3/plots/untrt/oh_rank_nn/2026-01-14_11-23" 
data = load(joinpath(base_path, "predstrues.jld2"))

all_trues = data["all_trues"]
all_preds = data["all_preds"]

cp = cor(Float64.(all_trues), Float64.(all_preds))
max_id = maximum(vcat(all_trues, all_preds))
bin_edges = 1:(max_id + 1) 

h = fit(Histogram, (all_trues, all_preds), (bin_edges, bin_edges))

begin
    fig_hm = Figure(size = (400, 300))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "True rank",
        ylabel = "Predicted rank",
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $(round(cp, digits=4))", color = :white)
    
    Colorbar(fig_hm[1, 2], hm, label = "Count (log10)")
    display(fig_hm)
end

save_dir = "/home/golem/scratch/chans/lincsv3/plots/untrt/oh_rank_nn/2026-01-14_11-23"
save(joinpath(save_dir, "heatmap.png"), fig_hm)