library(caret)
library(pROC)
library(parabar)
library(dplyr)
library(ggplot2)
library(clusterGeneration)

# Create mega-population function.

generate_mega_population <- function(n_obs = 100000, n_vars = 20, coef_mean = 0, coef_sd = 0.9) {
    # Generate independent predictors
    P <- matrix(rnorm(n_obs * n_vars), nrow = n_obs)
    colnames(P) <- paste0("X", 1:n_vars)


    # Generate coefficients (intercept = 0 for 50% prevalence)
    true_betas <- c(Intercept = 0, rnorm(n_vars, coef_mean, coef_sd))


    # Calculate probabilities
    linear_combo <- cbind(1, P) %*% true_betas
    probabilities <- plogis(linear_combo)


    # Generate balanced outcome
    y <- rbinom(n_obs, 1, prob = probabilities)

    # Transform y into a factor
    y <- factor(y, levels = c(0, 1), labels = c("No", "Yes"))


    return(list(
        data = data.frame(P, y),
        true_coefficients = true_betas
    ))
}





# Task functions.


#Logistic regression
task_logistic_regression <- function(sample_size) {

    # Simple sampling keeping balanced population
    index <- createDataPartition(data$y, p = sample_size / nrow(data), list = FALSE)
    sample_data <- data[index, ]


    # Split the sample data
    train_index <- createDataPartition(sample_data$y, p = 0.8, list = FALSE)
    # Create training and test set
    training_data <- sample_data[train_index, ]
    test_data <- sample_data[-train_index, ]

    # Model
    model <- train(y ~ .,
        data = training_data,
        method = "glm",
        family = "binomial",
        preProcess = c("nzv")
    )

    predictions <- predict(model, newdata = test_data, type = "prob")[, "Yes"]

    ## Convert y to numeric for ROC analysis
    test_data$y <- as.numeric(test_data$y)

    # Calculate AUC
    roc_obj <- roc(test_data$y, predictions, quiet = TRUE)

    # Return the AUC value.
    return(auc(roc_obj))

}

#Regularized regression
task_regularized_regression <- function(sample_size) {

    # Simple sampling keeping balanced population
    index <- createDataPartition(data$y, p = sample_size / nrow(data), list = FALSE)
    sample_data <- data[index, ]


    # Split the sample data
    train_index <- createDataPartition(sample_data$y, p = 0.8, list = FALSE)
    # Create training and test set
    training_data <- sample_data[train_index, ]
    test_data <- sample_data[-train_index, ]

    # Model
    model <- train(y ~ .,
        data = training_data,
        method = "glmnet",
        family = "binomial",
        tuneLenght = 10,
        preProcess = c("nzv")
    )

    predictions <- predict(model, newdata = test_data, type = "prob")[, "Yes"]

    ## Convert y to numeric for ROC analysis
    test_data$y <- as.numeric(test_data$y)

    # Calculate AUC
    roc_obj <- roc(test_data$y, predictions, quiet = TRUE)

    # Return the AUC value.
    return(auc(roc_obj))

}

#Linear Support Vector Machines
task_linear_support_vectors <- function(sample_size) {

    # Simple sampling keeping balanced population
    index <- createDataPartition(data$y, p = sample_size / nrow(data), list = FALSE)
    sample_data <- data[index, ]


    # Split the sample data
    train_index <- createDataPartition(sample_data$y, p = 0.8, list = FALSE)
    # Create training and test set
    training_data <- sample_data[train_index, ]
    test_data <- sample_data[-train_index, ]

    # Model
    model <- train(y ~ .,
        data = training_data,
        method = "svmLinear",
        tuneLength = 5,
        metric = "ROC",
        trControl = trainControl(
            method = "none",
            classProbs = TRUE,
            summaryFunction = twoClassSummary
        ),
        preProcess = c("nzv")
    )

    predictions <- predict(model, newdata = test_data, type = "prob")[, "Yes"]

    ## Convert y to numeric for ROC analysis
    #test_data$y <- as.numeric(test_data$y)

    # Calculate AUC
    roc_obj <- roc(response = test_data$y, predictor = predictions, quiet = TRUE)

    # Return the AUC value.
    return(auc(roc_obj))

}

#Random forests
task_random_forests <- function(sample_size) {

    # Simple sampling keeping balanced population
    index <- createDataPartition(data$y, p = sample_size / nrow(data), list = FALSE)
    sample_data <- data[index, ]


    # Split the sample data
    train_index <- createDataPartition(sample_data$y, p = 0.8, list = FALSE)
    # Create training and test set
    training_data <- sample_data[train_index, ]
    test_data <- sample_data[-train_index, ]

    # Model
    model <- train(y ~ .,
        data = training_data,
        method = "rf",
        #tuneLength = 5,
        trControl = trainControl(
            method = "none",
            classProbs = TRUE
        )
        #preProcess = c("nzv")
    )

    predictions <- predict(model, newdata = test_data, type = "prob")[, "Yes"]

    ## Convert y to numeric for ROC analysis
    #test_data$y <- as.numeric(test_data$y)

    # Calculate AUC
    roc_obj <- roc(test_data$y, predictions, quiet = TRUE)

    # Return the AUC value.
    return(auc(roc_obj))

}

has_converged_consecutive <- function(results_df, pop_auc_value, convergence_threshold, consecutive_n = 5) {
  # Calculate the absolute difference between mean AUC and population AUC
  results_df$auc_dif <- abs(results_df$mean_auc - pop_auc_value)

  # Check if the last 'consecutive_n' auc_dif values are below the threshold
  if (nrow(results_df) >= consecutive_n) {
    last_n_diffs <- tail(results_df$auc_dif, consecutive_n)
    if (all(last_n_diffs < convergence_threshold)) {
      # Find the sample size of the first of these consecutive values
      converged_sample_size <- results_df$sample_size[nrow(results_df) - consecutive_n + 1]
      return(converged_sample_size)
    }
  }
  # Not converged yet
  return(NULL)
}

find_convergence_sample_size <- function(data, task, pop_auc_value, start_size = 100, end_size = nrow(data), step_size = 100, repetitions = 30, convergence_threshold = 0.1, n_cores = 1) {

  # Start the backend.
  backend <- start_backend(n_cores, "psock", "async")

  # Close the backend on exit.
  on.exit(stop_backend(backend))

  # Load packages on the backend.
  evaluate(backend, {
    library(caret)
    library(pROC)
  })

  # Export the data to the cluster from the function environment.
  export(backend, "data", environment())

  # Set the progress bar format.
  configure_bar(type = "modern", format = "[:bar] :percent")

  sample_sizes <- seq(start_size, end_size, step_size)
  results_df <- data.frame(sample_size = integer(), mean_auc = numeric(), CI_lower = numeric(), CI_upper = numeric(), stringsAsFactors = FALSE)

  for (current_size_index in seq_along(sample_sizes)) {
    current_size <- sample_sizes[current_size_index]

    configure_bar(type = "modern", format = paste0("Sample: ", current_size, " | [:bar] :percent"))

    # Run for the current sample size.
    auc_values <- par_sapply(backend, rep(current_size, repetitions), task)

    # Calculate sample auc difference to population auc
    mean_auc <- mean(auc_values)
    se_auc <- sd(auc_values) / sqrt(repetitions)
    CI_lower <- mean_auc - 1.96 * se_auc
    CI_upper <- mean_auc + 1.96 * se_auc

    # Store the results for the current sample size
    results_df <- rbind(results_df, data.frame(sample_size = current_size, mean_auc = mean_auc, CI_lower = CI_lower, CI_upper = CI_upper))

    # Stopping rule: Check for convergence using the external function
    converged_at <- has_converged_consecutive(results_df, pop_auc_value, convergence_threshold, consecutive_n = 5)

    if (!is.null(converged_at)) {
      return(list(converged_at_n = converged_at, results = results_df))
    }

    # If we've reached the maximum sample size and haven't converged
    if (current_size == end_size) {
      return(list(converged_at_n = NULL, results = results_df, message = paste0("More than ", end_size, " observations are needed to reach reliable estimates.")))
    }
  }
}

#Functions classification model for the full population (includes cross-validation)
logistic_regression_population <- function(train_data, test_data){
    my_model <- train(y ~ .,

    data = train_data,
    method = "glm",
    family = "binomial",

    #Cross validation
    trControl = trainControl(
        method = "cv",
        number = 5

    ),

    preProcess = c("nzv")
    )


    # Make predictions on test data
    predictions <- predict(my_model, newdata = test_data, type = "prob")

    # Calculate AUC on test data
    roc_obj <- roc(test_data$y, predictions$"Yes", quiet = TRUE)
    pop_auc <- auc(roc_obj)
    return(pop_auc)
}

regularized_logistic_regression_population <- function(train_data, test_data){
    my_model <- train(y ~ .,

    data = train_data,
    method = "glmnet",
    family = "binomial",

    #Cross validation
    trControl = trainControl(
        method = "cv",
        number = 5

    ),
    tuneLenght = 10,
    preProcess = c("nzv")
    )


    # Make predictions on test data
    predictions <- predict(my_model, newdata = test_data, type = "prob")

    # Calculate AUC on test data
    roc_obj <- roc(test_data$y, predictions$"Yes", quiet = TRUE)
    pop_auc <- auc(roc_obj)
    return(pop_auc)
}

linear_support_vectors_population <- function(train_data, test_data){
    my_model <- train(y ~ .,

    data = train_data,
    method = "svmLinear",
    metric = "ROC",

    #Cross validation
    trControl = trainControl(
        method = "cv",
        number = 5,
        classProbs = TRUE,
        #Model performance based on AUC
        summaryFunction = twoClassSummary
    ),

    tuneLenght = 5,
    preProcess = c("nzv")
    )


    # Make predictions on test data
    predictions <- predict(my_model, newdata = test_data, type = "prob")

    # Calculate AUC on test data
    roc_obj <- roc(test_data$y, predictions$"Yes", quiet = TRUE)
    pop_auc <- auc(roc_obj)
    return(pop_auc)
}

random_forest_population <- function(train_data, test_data){
    my_model <- train(y ~ .,

    data = train_data,
    method = "rf",

    #Cross validation
    trControl = trainControl(
        method = "cv",
        number = 5
    ),

    tuneLenght = 5,
    preProcess = c("nzv")
    )


    # Make predictions on test data
    predictions <- predict(my_model, newdata = test_data, type = "prob")

    # Calculate AUC on test data
    roc_obj <- roc(test_data$y, predictions$"Yes", quiet = TRUE)
    pop_auc <- auc(roc_obj)
    return(pop_auc)
}

#Plotting
sample_plot <- function(results_df, pop_auc_value,convergence_threshold = 0.01, converged_at_n) {

  # Calculate AUC deviation
  results_df$auc_deviation <- results_df$mean_auc - pop_auc_value

  # Create the plot
  p <- ggplot(results_df, aes(x = sample_size, y = auc_deviation)) +
    geom_line() +
    # Confidence interval ribbon
    geom_ribbon(aes(ymin = (CI_lower - pop_auc_value), ymax = (CI_upper - pop_auc_value)), alpha = 0.2, fill = "blue") +
    # Line at 0 to show population AUC
    geom_hline(yintercept = 0, color = "pink", linetype = "solid", size = 2) +
    # Convergence threshold line
    geom_hline(yintercept = convergence_threshold, color = "red", linetype = "dashed", size = 1) +
    geom_hline(yintercept = -convergence_threshold, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = converged_at_n, color = "blue", linetype = "solid", size = 2)+
    labs(
      x = "Sample Size",
      y = "Deviation from AUC population performance",
      title = "Convergence of Sample AUC to Population AUC"
    ) +
    theme_minimal() +
    theme(
      # Center the title
      plot.title = element_text(hjust = 0.5),
      #Center the label of the x-axis
      axis.title.x = element_text(angle = 0, hjust = 0.5, vjust=0.5),
      #Rotate the labels
      axis.text.x = element_text(angle = 45, hjust = 1)
    )+
    scale_x_continuous(breaks = unique(results_df$sample_size))

#Vertical line at the recommended sample size only if converged_at_n is not NULL
if (!is.na(converged_at_n) && !is.null(converged_at_n)) {
  p <- p + geom_vline(xintercept = converged_at_n, color = "blue", linetype = "solid", size = 2)
}

  return(p)
}

#Wrapper function
recommend_sample_size <- function(n_obs = 100000,
                                  n_vars,
                                  coef_mean = 0,
                                  coef_sd = 0.9,
                                  model_type,
                                  start_size = 300,
                                  end_size = 10000,
                                  step_size = 100,
                                  repetitions = 30,
                                  convergence_threshold = 0.01,
                                  n_cores = 7
                                  ){
        print(paste0("Generating mega-population with ", n_obs, " observations and ", n_vars, " variables."))
        # Generate the mega-population
        mega_population <- generate_mega_population(n_obs = n_obs, n_vars = n_vars, coef_mean = coef_mean, coef_sd = coef_sd)

        # Split the data in training and test sets
        train_index <- createDataPartition(mega_population$data$y, p = 0.8, list = FALSE)
        pop_train_data <- mega_population$data[train_index, ]
        pop_test_data <- mega_population$data[-train_index, ]

        print(paste0("Fitting classification model: ", model_type, " on the mega-population."))
        # Fit classification model and get the population AUC value
        if(model_type == "logistic regression")
            auc_my_model <- logistic_regression_population(train_data = pop_train_data, test_data = pop_test_data)
        if(model_type == "regularized logistic regression")
            auc_my_model <- regularized_logistic_regression_population(train_data = pop_train_data, test_data = pop_test_data)
        if(model_type == "linear support vector machines")
            auc_my_model <- linear_support_vectors_population(train_data = pop_train_data, test_data = pop_test_data)
        if(model_type == "random forests")
            auc_my_model <- random_forest_population(train_data = pop_train_data, test_data = pop_test_data)

        # Print the population AUC value
        print(paste0("Population AUC value: ", auc_my_model))

        # Find the sample size for convergence
        recommendation <- find_convergence_sample_size(
            mega_population$data,
            pop_auc_value = auc_my_model,
            start_size = start_size,
            end_size = end_size,
            step_size = step_size,
            repetitions = repetitions,
            convergence_threshold = convergence_threshold,
            task = match.fun(paste0("task_", gsub(" ", "_", model_type))),
            n_cores = n_cores
        )

   #Return the recommendation and plot only if converged
   if (!is.null(recommendation$converged_at_n)){
    plot <- sample_plot(recommendation$results, pop_auc_value = auc_my_model, converged_at_n = recommendation$converged_at_n)
    print(plot)}
  return(recommendation)

}

# Example usage of the wrapper function
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

