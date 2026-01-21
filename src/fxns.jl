using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using Flux, CUDA, Random, OneHotArrays

function rank_genes(expr, medians)
    n, m = size(expr)
    data_ranked = Matrix{Int}(undef, size(expr)) # faster than fill(-1, size(expr))
    normalized_col = Vector{Float32}(undef, n) 
    sorted_ind_col = Vector{Int}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        @. normalized_col = unsorted_expr_col / medians
        sortperm!(sorted_ind_col, unsorted_expr_col, rev=true)
            # rev=true -> data[1, :] = index (into gene.expr) of highest expression value in experiment/column 1
        for i in 1:n
            data_ranked[i, j] = sorted_ind_col[i]
        end
    end
    return data_ranked
end

function split_data(X, test_ratio::Float64, y=nothing) # masking doesn't need y!
    n = size(X, 2)
    indices = shuffle(1:n)
    test_size = floor(Int, n * test_ratio)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    X_train = X[:, train_indices]
    X_test = X[:, test_indices]
    if y === nothing
        return X_train, X_test, train_indices, test_indices
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test, train_indices, test_indices
    end
end

function mask_input(X::Matrix, mask_ratio::Float64, mask, mask_id)
    X_masked = copy(X)
    mask_labels = fill((mask), size(X)) # -100 = ignore, this is not masked
    
    for j in 1:size(X,2) # per column, start at second row so we don't mask CLS token
        num_masked = ceil(Int, (size(X,1) - 1) * mask_ratio)

        if typeof(mask_id) == Int64
            mask_positions_local = randperm(size(X,1) - 1)[1:num_masked]
            mask_positions_global = mask_positions_local .+ 1 # also shifted for CLS token
        elseif typeof(MASK_VALUE) == Float32
            mask_positions_global = randperm(size(X,1))[1:num_masked]
        end

        for pos in mask_positions_global
            mask_labels[pos, j] = X[pos, j] # original label
            X_masked[pos, j] = mask_id # mask label
        end
    end
    return X_masked, mask_labels
end

function get_rank(values, ref)
    combined = vcat(values, ref)
    p = sortperm(combined, rev=true)
    ranks = invperm(p)
    return ranks[1]
end

function convert_exp_to_rank(X_test, all_trues, all_preds)
    reference_matrix = X_test 
    ranked_preds = similar(all_preds, Int)
    ranked_trues = similar(all_trues, Int)

    for i in 1:length(all_preds)
        pred = all_preds[i]
        true_val = all_trues[i]
        col_idx = all_column_indices[i]
        reference_col = reference_matrix[:, col_idx]
        ranked_preds[i] = get_rank(pred, reference_col)
        ranked_trues[i] = get_rank(true_val, reference_col)
    end
    return ranked_trues, ranked_preds
end