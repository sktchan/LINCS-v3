using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using JLD2, Dates, DataFrames, Statistics, CUDA, Flux


function save_exp(model, train_losses, test_losses, X_test_masked, y_test_masked, X_test, 
                  test_indices, train_indices, plot_results, start_time, save_dir, n_epochs)
    
    model_cpu = cpu(model)
    model_state = Flux.state(model_cpu)
    jldsave(joinpath(save_dir, "model_state.jld2"); 
        model_state=model_state
    )
    jldsave(joinpath(save_dir, "model_object.jld2"); 
        model=model_cpu
    )
    jldsave(joinpath(save_dir, "losses.jld2"); 
        epochs = 1:n_epochs, 
        train_losses = train_losses, 
        test_losses = test_losses
    )
    jldsave(joinpath(save_dir, "predstrues.jld2"); 
        all_preds = plot_results.all_preds, 
        all_trues = plot_results.all_trues,
        all_gene_indices = plot_results.all_gene_indices
    )
    jldsave(joinpath(save_dir, "indices.jld2"); 
        test_indices = test_indices, 
        train_indices = train_indices
    )
    jldsave(joinpath(save_dir, "masked_test_data.jld2"); X=X_test_masked, y=y_test_masked)
    jldsave(joinpath(save_dir, "test_data.jld2"); X=X_test)

    end_time = now()
    run_time = end_time - start_time
    total_minutes = div(run_time.value, 60000)
    run_hours = div(total_minutes, 60)
    run_minutes = rem(total_minutes, 60)

    params_txt = joinpath(save_dir, "params.txt")
    open(params_txt, "w") do io
        println(io, "PARAMETERS:")
        println(io, "########### $(gpu_info)")
        println(io, "dataset = $(dataset)")
        println(io, "masking_ratio = $mask_ratio")
        println(io, "mask_value = $MASK_VALUE")
        println(io, "NO DYNAMIC MASKING")
        println(io, "batch_size = $batch_size")
        println(io, "n_epochs = $n_epochs")
        println(io, "embed_dim = $embed_dim")
        println(io, "hidden_dim = $hidden_dim")
        println(io, "n_heads = $n_heads")
        println(io, "n_layers = $n_layers")
        println(io, "learning_rate = $lr")
        println(io, "dropout_probability = $drop_prob")
        println(io, "ADDITIONAL NOTES: $(additional_notes)")
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
        println(io, "correlation = $(plot_results.correlation)")
        println(io, "mse model = $(plot_results.mse_model)")
        println(io, "mse baseline = $(plot_results.mse_baseline)")
    end
end


function save_rank(model, train_losses, test_losses, X, X_test_masked, y_test_masked, 
                  test_indices, train_indices, plot_results, start_time, save_dir, 
                  n_epochs, batch_size)
    
    model_cpu = cpu(model)
    model_state = Flux.state(model_cpu)

    function get_profile_embedding(m, input)
        transformed = m.transformer(m.pos_dropout(m.pos_encoder(m.embedding(input))))
        return transformed[:, 1, :] # Assuming [CLS] token is at position 1
    end

    all_embeddings = []
    Flux.testmode!(model)
    for start_idx in 1:batch_size:size(X, 2)
        end_idx = min(start_idx + batch_size - 1, size(X, 2))
        input_batch = gpu(X[:, start_idx:end_idx])
        batch_embeddings = cpu(get_profile_embedding(model, input_batch))
        push!(all_embeddings, batch_embeddings)
    end
    final_embeddings = hcat(all_embeddings...) 

    jldsave(joinpath(save_dir, "model_state.jld2"); 
        model_state=model_state
    )
    jldsave(joinpath(save_dir, "model_object.jld2"); 
        model=model_cpu
    )
    jldsave(joinpath(save_dir, "indices.jld2"); 
        train_indices=train_indices, 
        test_indices=test_indices
    )
    jldsave(joinpath(save_dir, "profile_embeddings.jld2"); 
        profile_embeddings=final_embeddings
    )
    jldsave(joinpath(save_dir, "losses.jld2"); 
        epochs = 1:n_epochs, 
        train_losses = train_losses, 
        test_losses = test_losses
    )
    jldsave(joinpath(save_dir, "predstrues.jld2"); 
        all_preds = plot_results.all_preds, 
        all_trues = plot_results.all_trues
    )
    jldsave(joinpath(save_dir, "rank_vs_error.jld2"); 
        original_rank = plot_results.all_original_ranks, 
        prediction_error = plot_results.all_prediction_errors
    )
    jldsave(joinpath(save_dir, "avg_errors.jld2"); 
        original_rank = plot_results.avg_errors.original_rank,
        avg_error = plot_results.avg_errors.avg_error
    )

    end_time = now()
    run_time = end_time - start_time
    total_minutes = div(run_time.value, 60000)
    run_hours = div(total_minutes, 60)
    run_minutes = rem(total_minutes, 60)

    params_txt = joinpath(save_dir, "params.txt")
    open(params_txt, "w") do io
        println(io, "PARAMETERS:")
        println(io, "########### $(gpu_info)")
        println(io, "dataset = $(dataset)")
        println(io, "masking_ratio = $mask_ratio")
        println(io, "NO DYNAMIC MASKING")
        println(io, "batch_size = $batch_size")
        println(io, "n_epochs = $n_epochs")
        println(io, "embed_dim = $embed_dim")
        println(io, "hidden_dim = $hidden_dim")
        println(io, "n_heads = $n_heads")
        println(io, "n_layers = $n_layers")
        println(io, "learning_rate = $lr")
        println(io, "dropout_probability = $drop_prob")
        println(io, "ADDITIONAL NOTES: $(additional_notes)")
        println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    end
end
