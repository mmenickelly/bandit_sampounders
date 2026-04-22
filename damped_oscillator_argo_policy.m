function probs = damped_oscillator_argo_policy(models, batch_size, sample_type, data)

    global max_t

    X_inc = data.X_inc;
    if strcmp(sample_type, 'model')
        already_updated = data.already_updated;
    elseif strcmp(sample_type, 'rho')
        Xsp = data.Xsp;
    end

    if strcmp(sample_type, 'model')
        prompt = write_prompt_for_damped_oscillator(X_inc, models, 0, max_t);
    elseif strcmp(sample_type, 'rho')
        prompt = write_prompt_for_damped_oscillator(Xsp, models, 0, max_t);
    end

    response = argoGeminiGenerateContent(prompt);
    % clean up for decode
    response_str = response.Body.Data.response;
    response_length = length(response_str);
    response_str = response_str(9:(response_length-3));
    response_json = jsondecode(response_str); 
    probs = response_json.array; 
    reasoning = response_json.reasoning; % nice to have, not saving for now. 

    % a practically useful masking:
    if strcmp(sample_type, 'model')
        probs(already_updated) = 0;
    end

    % clean up in case LLM screwed up the assignment:
    probs = batch_size * probs / sum(probs);

    if length(probs) > length(models)
        probs = probs(1:length(models));
    end

end
