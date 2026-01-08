using ProgressBars, CUDA, Flux, Statistics

function train(model, opt, n_epochs, batch_size, X_train_masked, y_train_masked, X_test_masked, y_test_masked)
    train_losses = Float32[]
    test_losses = Float32[]
    test_rank_errors = Float32[]
    is_rank_model = typeof(model) == RankModel

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

            # Compute rank errors for rank models
            if is_rank_model && !isempty(y_masked)
                logits_cpu = cpu(logits_masked)
                y_cpu = cpu(y_masked)
                
                for i in 1:length(y_cpu)
                    true_gene_id = y_cpu[i]
                    prediction_logits = logits_cpu[:, i]
                    ranked_gene_ids = sortperm(prediction_logits, rev=true)
                    predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids)
                    
                    if !isnothing(predicted_rank)
                        error = predicted_rank - 1
                        push!(epoch_rank_errors, error)
                    end
                end
            end
        end

        push!(test_losses, mean(test_epoch_losses))
        if is_rank_model
            if !isempty(epoch_rank_errors)
                push!(test_rank_errors, mean(epoch_rank_errors))
            else
                push!(test_rank_errors, NaN32)
            end
        end
    end

    if is_rank_model
        return train_losses, test_losses, test_rank_errors
    else
        return train_losses, test_losses
    end
end

function test()
end