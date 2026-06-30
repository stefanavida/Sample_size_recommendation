# Sample_size_recommendation

Wrapper function _recommend_sample_size()_   recommends a minimum training sample size for popular binary classification models based on simulations. The core idea: simulate a large "mega-population", measure the predictive performance (AUC) achievable on it, then determine how many observations a study actually needs before its sample-level performance converges to that population ceiling.

**Subfunctions: **
* _generate_mega_population_ : Generates a mega-population of correlated predictors and a balanced binary outcome from a logistic data-generating model, with user control over the number of variables, signal strength (coefficient mean/SD), and inter-variable correlation (dependencies).
* _find_convergence_sample_size_ : Searches for convergence for a linearly increasing sequence of candidate sample sizes, it repeatedly draws samples, trains the model, and records the mean test AUC plus a 95% confidence interval. Sampling runs in parallel across cores via the 'parabar' R package.
* _has_converged_consecutive_ : stopping rule (AUC stays within 'convergence_threshold' of the population AUC for 5 consecutive steps). Convergence threshold is set at a default of 0.01 which translates at a 1% difference in prediction performance compared to the population, but it can be changed by the user for more conversative scenarios. 
* plotting

** Suported models **
* Logistic regression (glm)
* Regularized logistic regression (glmnet)
* Linear support vector machines (svmLinear)
* Random forests (rf)

** Example usage of the wrapper function **
```r
set.seed(20011001)
recommendation <- recommend_sample_size(
    n_obs = 100000, # Population size
    n_vars = 13, # Number of variables in the mega-population
    coef_mean = 0.2, # Mean of the coefficients
    coef_sd = 0.3, # Standard deviation of the coefficients
    model_type = "logistic regression",
    start_size = 100, # Min sample size range
    end_size = 2500, # Max sample size range (budget constrains)
    step_size = 100,
    repetitions = 10,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation
```

