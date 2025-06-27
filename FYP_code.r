library(caret)
library(pROC)
library(parabar)
library(dplyr)
library(ggplot2)
library(clusterGeneration)
library(mvtnorm)
library(data.table)
library(plotly)

# Create mega-population function.

generate_mega_population <- function(n_obs = 100000, n_vars = 20, coef_mean = 0, coef_sd = 0.9, dependencies = 0.2) {

    #Generate a correlation matrix with values between -dependencies and +dependencies
    cor_matrix <- matrix(runif(n_vars * n_vars, -dependencies, dependencies), nrow = n_vars, ncol = n_vars)
    # Ensure the diagonal is 1
    diag(cor_matrix) <- 1
    # Make the matrix symmetric
    cor_matrix[lower.tri(cor_matrix)] <- t(cor_matrix)[lower.tri(cor_matrix)]


    # Generate independent predictors
    #P <- matrix(rnorm(n_obs * n_vars), nrow = n_obs)
    P <- rmvnorm(n = n_obs, mean = rep(0, n_vars), sigma = cor_matrix)
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
task_regularized_logistic_regression <- function(sample_size) {

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
task_linear_support_vector_machines <- function(sample_size) {

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
                                  dependencies = 0.2, #correlation between variables
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
        mega_population <- generate_mega_population(n_obs = n_obs, n_vars = n_vars, coef_mean = coef_mean, coef_sd = coef_sd, dependencies = dependencies)
        print("Mega-population generated.")
        # Print the first few rows of the mega-population data
        print(head(mega_population$data))
        print("Fitting classification model on the mega-population to get the population AUC value.")
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


# # Simulation studies
set.seed(20011001)
# Parameters that will be varied in the simulations
MODEL_TYPES <- c("logistic regression",
                  "regularized logistic regression",
                 "linear support vector machines",
                 "random forests")



#Scenario 1: Varying coefficient means
COEF_MEAN_VALUES <- c(0.1, 0.2, 0.3, 0.4, 0.5)
# Create a list to store results from each run
results_scenario1_list <- list()
count <- 0
# Loop through each model type and each value of coef_mean
cat("Running Scenario 1: Varying coef_mean...\n")
for (model in MODEL_TYPES) {
    for (c_mean in COEF_MEAN_VALUES) {

        count <- count + 1
        # Print progress to the console
        cat(paste("  Model:", model, "| Coef Mean:", c_mean, "\n"))

        # Call your function
        recommendation_output <- recommend_sample_size(
            n_obs = 100000,
            n_vars = 10,
            coef_mean = c_mean,
            coef_sd = 0.2,
            model_type = model,
            start_size = 100,
            end_size = 5000,
            step_size = 100,
            repetitions = 15,
            convergence_threshold = 0.01,
            n_cores = 7
        )
        results_scenario1_list[[count]] <- data.table(
            number_of_variables = 100000,
            coefficient_mean = c_mean,
            model_type = model,
            recommendation = ifelse(is.numeric(recommendation_output$converged_at_n) == FALSE, "***", recommendation_output$converged_at_n)
            )
    }
}

#Convert list into a dataframe
results_scenario1_all_models <- rbindlist(results_scenario1_list)
#Visualise the results
head(results_scenario1_all_models)

#Plotting the simulation results (scenario 1)
ggplot(results_scenario1_all_models, aes(x = coefficient_mean, y = recommendation, color = model_type)) +
 geom_point(size = 4, alpha = 0.7)+
  labs(
    title = "Recommendation sample size vs. Coefficient Mean by Model Type",
    x = "Coefficient Mean",
    y = "Minimum number of observations",
    color = "Model Type"
  ) +
  theme_minimal()+
  theme(
    # Legend Text (the individual items/levels in the legend)
    legend.text = element_text(size = 16), # Adjust size as needed

    # Legend Title (e.g., "Model Type")
    legend.title = element_text(size = 16, face = "bold"), # Adjust size and font face

    # Axis Titles (x and y labels, e.g., "Coefficient Mean", "Recommendation")
    axis.title = element_text(size = 14, face = "bold"), # Adjust size and font face

    # Axis Tick Labels (the numbers or categories on the axes)
    axis.text = element_text(size = 12) # Adjust size as needed
  )



#Scenario 2: Varying number of predictors
N_VARS_VALUES <- c( 10, 15, 20, 30)
# Create a list to store results from each run
results_scenario2_list <- list()
count <- 0
# Loop through each model type and each value of coef_mean
cat("Running Scenario 2: Varying number of variables...\n")
for (model in MODEL_TYPES) {
    for (nr_vars in N_VARS_VALUES) {

        count <- count + 1
        # Print progress to the console
        cat(paste("  Model:", model, "| Number predictors:", nr_vars, "\n"))

        # Call your function
        recommendation_output <- recommend_sample_size(
            n_obs = 100000,
            n_vars = nr_vars,
            coef_mean = 0.3,
            coef_sd = 0.2,
            dependencies = 0,
            model_type = model,
            start_size = 100,
            end_size = 5000,
            step_size = 100,
            repetitions = 15,
            convergence_threshold = 0.01,
            n_cores = 7
        )
        print(recommendation_output)
        # Store the results in a list
        results_scenario2_list[[count]] <- data.table(
            number_of_predictors = nr_vars,
            model_type = model,
            recommendation = ifelse(is.numeric(recommendation_output$converged_at_n) == FALSE, "***", recommendation_output$converged_at_n)
            )
    }
}

sol_linear_support_vector_machines <- results_scenario2_support_vector[model_type == "linear support vector machines"]
sol_regularized_logistic_regression <- results_scenario2_all_models[model_type == "regularized logistic regression"]
sol_logistic_regression <- results_scenario2_all_models[model_type == "logistic regression"]
sol_random_forests <- results_scenario2_random_forests

results_scenario2_all_models <- rbind(
    sol_logistic_regression,
    sol_regularized_logistic_regression,
    sol_linear_support_vector_machines,
    sol_random_forests
)

#Plotting the simulation results (scenario 2)
ggplot(results_scenario2_all_models, aes(x = number_of_predictors, y = recommendation, color = model_type)) +
 geom_point(size = 4, alpha = 0.7)+
  labs(
    title = "Recommendation sample size vs. Number of predictors by Model Type",
    x = "Number of predictors",
    y = "Minimum number of observations",
    color = "Model Type"
  ) +
  theme_minimal()+
  theme(
    # Legend Text (the individual items/levels in the legend)
    legend.text = element_text(size = 16), # Adjust size as needed

    # Legend Title (e.g., "Model Type")
    legend.title = element_text(size = 16, face = "bold"), # Adjust size and font face

    # Axis Titles (x and y labels, e.g., "Coefficient Mean", "Recommendation")
    axis.title = element_text(size = 14, face = "bold"), # Adjust size and font face

    # Axis Tick Labels (the numbers or categories on the axes)
    axis.text = element_text(size = 12) # Adjust size as needed
  )

# SIMULATION 3
COEF_MEAN_VALUES <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
N_VARS_VALUES <- c(10, 15, 20, 25, 30)

# Create a timestamp for unique filenames
timestamp_str <- format(Sys.time(), "%Y%m%d_%H%M%S")
rdata_filepath <- paste0("simulation_results_", timestamp_str, ".Rdata")
excel_filepath <- paste0("simulation_results_", timestamp_str, ".xlsx")
error_log_filepath <- paste0("simulation_errors_", timestamp_str, ".log") # Don't forget this one!

# Initialize an empty list to store results
results_scenario3_list <- list()
count <- 0 # Initialize counter for results list

# Define how often to save intermediate results (e.g., every 5 iterations)
SAVE_INTERVAL <- 5

cat("Running Scenario: Varying number of predictors and coefficient mean...\n")

# Open a connection to the error log file
# This should now use the uniquely named error_log_filepath
error_log_con <- file(error_log_filepath, open = "a")

for (coef_mean_val in COEF_MEAN_VALUES) { # Renamed 'coef_mean' to 'coef_mean_val' to avoid conflict with function arg
  for (nr_vars_val in N_VARS_VALUES) { # Renamed 'nr_vars' to 'nr_vars_val'
    count <- count + 1

    # Print progress to the console
    cat(paste0("  Processing (", count, "/", length(COEF_MEAN_VALUES) * length(N_VARS_VALUES), ") - Coef Mean: ", coef_mean_val, " | Number predictors: ", nr_vars_val, "\n"))

    # Use tryCatch to handle potential errors
    result <- tryCatch({
      # Call your function with the appropriate arguments
      recommendation_output <- recommend_sample_size(
        n_obs = 100000,
        n_vars = nr_vars_val,
        coef_mean = coef_mean_val,
        coef_sd = 0.2,
        dependencies = 0.05,
        model_type = "logistic regression",
        start_size = 300,
        end_size = 5000,
        step_size = 100,
        repetitions = 10,
        convergence_threshold = 0.01,
        n_cores = 7
      )

      # --- IMPORTANT: Accessing the 'converged_at_n' element from the returned list ---
      # recommendation_output is a list, we need to extract its 'converged_at_n' component.
      # Your original line: recommendation = ifelse(is.numeric(recommendation_output$converged_at_n) == FALSE, "***", recommendation_output$converged_at_n)
      # This is correct. Let's make sure it handles NULL too if 'converged_at_n' isn't set.

      rec_value_raw <- recommendation_output$converged_at_n

      rec_value <- ifelse(
        is.null(rec_value_raw) || is.character(rec_value_raw), # Check for NULL or character string
        "***",
        rec_value_raw
      )

      # Return the data.table for this iteration
      data.table(
        number_of_predictors = nr_vars_val,
        coef_mean = coef_mean_val,
        recommendation = rec_value
      )

    }, error = function(e) {
      # Error handling block
      error_message <- paste0(
        format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"),
        " Error at Coef Mean: ", coef_mean_val,
        " | Number predictors: ", nr_vars_val,
        " - Error: ", e$message, "\n"
      )
      cat(error_message, file = error_log_con) # Write to log file
      warning(error_message) # Also print as a warning in console

      # Return a data.table with "ERROR" for this iteration
      data.table(
        number_of_predictors = nr_vars_val,
        coef_mean = coef_mean_val,
        recommendation = "ERROR" # Mark as error
      )
    })

    # Store the results (whether success or error)
    results_scenario3_list[[count]] <- result

    # Save intermediate results periodically
    # Also save on the very last iteration
    total_iterations <- length(COEF_MEAN_VALUES) * length(N_VARS_VALUES)
    if (count %% SAVE_INTERVAL == 0 || count == total_iterations) {
      cat(paste0("Saving intermediate results at iteration ", count, "...\n"))

      # Convert list of data.tables to a single data.table for saving
      current_df <- rbindlist(results_scenario3_list, fill = TRUE)

      # Save to Rdata
      save(current_df, file = rdata_filepath)

      # Save to Excel
      tryCatch({
        write_xlsx(current_df, path = excel_filepath)
      }, error = function(e) {
        warning(paste("Could not save to Excel:", e$message))
        cat(paste("Excel Save Error: ", e$message, "\n"), file = error_log_con)
      })
    }
  }
}

# Close the error log file connection
close(error_log_con)
cat("\nSimulation complete!\n")
cat(paste("Final results saved to:", rdata_filepath, " and ", excel_filepath, "\n"))
cat(paste("Errors (if any) logged to:", error_log_filepath, "\n"))

# Final combined dataframe
final_df_scenario3 <- rbindlist(results_scenario3_list, fill = TRUE)
print("Final Simulation Results:")
print(final_df_scenario3)


#Plot the results
plot_df <- copy(final_df_scenario3) # Create a copy to avoid modifying the original
plot_df[, recommendation_numeric := as.numeric(ifelse(recommendation == "***" | recommendation == "ERROR", NA, recommendation))]

# Convert 'number_of_predictors' to a factor for discrete coloring
plot_df[, number_of_predictors_factor := factor(number_of_predictors)]

ggplot(plot_df, aes(x = coef_mean, y = recommendation_numeric, color = number_of_predictors_factor)) +
  geom_point(size = 4, alpha = 0.7) +
  labs(
    #title = "Recommended Sample Size vs. Effect Strength by Number of Predictors",
    x = "Signal Strength",
    y = "Minimum Number of Observations",
    color = "No. of Predictors" # Legend title
  ) +
  theme_minimal() +
  theme(
    # Legend Text (the individual items/levels in the legend)
    legend.text = element_text(size = 12), # Adjust size as needed, slightly smaller than title

    # Legend Title (e.g., "Model Type")
    legend.title = element_text(size = 14, face = "bold"), # Adjust size and font face

    # Axis Titles (x and y labels, e.g., "Coefficient Mean", "Recommendation")
    axis.title = element_text(size = 14, face = "bold"), # Adjust size and font face

    # Axis Tick Labels (the numbers or categories on the axes)
    axis.text = element_text(size = 12), # Adjust size as needed

    # Plot Title
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5) # Center title and adjust size
  )




# SIMULATION 4: Varying signal strength and dependencies
COEF_MEAN_VALUES <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEPENDENCIES_VALUES <- c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5) # Correlations between variables
CONSTANT_N_VARS <- 10
CONSTANT_MODEL_TYPE <- "logistic regression"

# --- File Paths for Saving ---
timestamp_str <- format(Sys.time(), "%Y%m%d_%H%M%S")
rdata_filepath <- paste0("simulation4_results_", timestamp_str, ".Rdata")
excel_filepath <- paste0("simulation4_results_", timestamp_str, ".xlsx")
error_log_filepath <- paste0("simulation4_errors_", timestamp_str, ".log")

# Initialize an empty list to store results
results_scenario4_list <- list()
count <- 0 # Initialize counter for results list

# Define how often to save intermediate results (e.g., every 5 iterations)
SAVE_INTERVAL <- 5

cat("Running Simulation 4: Varying signal strength and dependencies (N_VARS = 10, Model = Logistic Regression)...\n")

# Open a connection to the error log file
error_log_con <- file(error_log_filepath, open = "a")

# Main simulation loop
for (coef_mean_val in COEF_MEAN_VALUES) {
    for (deps_val in DEPENDENCIES_VALUES) {
        count <- count + 1

        # Print progress to the console
        cat(paste0("  Processing (", count, "/", length(COEF_MEAN_VALUES) * length(DEPENDENCIES_VALUES), ") - Coef Mean: ", coef_mean_val, " | Dependencies: ", deps_val, "\n"))

        # Use tryCatch to handle potential errors
        result <- tryCatch({
            recommendation_output <- recommend_sample_size(
                n_obs = 100000,
                n_vars = CONSTANT_N_VARS,
                coef_mean = coef_mean_val,
                coef_sd = 0.2, # Keeping consistent with prior examples
                dependencies = deps_val,
                model_type = CONSTANT_MODEL_TYPE,
                start_size = 300,
                end_size = 10000,
                step_size = 100,
                repetitions = 30,
                convergence_threshold = 0.01,
                n_cores = 7
            )

            rec_value_raw <- recommendation_output$converged_at_n

            rec_value <- ifelse(
                is.null(rec_value_raw) || is.character(rec_value_raw),
                "***",
                rec_value_raw
            )

            data.table(
                coef_mean = coef_mean_val,
                dependencies = deps_val,
                recommendation = rec_value
            )

        }, error = function(e) {
            error_message <- paste0(
                format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"),
                " Error at Coef Mean: ", coef_mean_val,
                " | Dependencies: ", deps_val,
                " - Error: ", e$message, "\n"
            )
            cat(error_message, file = error_log_con)
            warning(error_message)

            data.table(
                coef_mean = coef_mean_val,
                dependencies = deps_val,
                recommendation = "ERROR"
            )
        })

        results_scenario4_list[[count]] <- result

        total_iterations <- length(COEF_MEAN_VALUES) * length(DEPENDENCIES_VALUES)
        if (count %% SAVE_INTERVAL == 0 || count == total_iterations) {
            cat(paste0("Saving intermediate results at iteration ", count, "...\n"))
            current_df <- rbindlist(results_scenario4_list, fill = TRUE)
            save(current_df, file = rdata_filepath)
            tryCatch({
                write_xlsx(current_df, path = excel_filepath)
            }, error = function(e) {
                warning(paste("Could not save to Excel:", e$message))
                cat(paste("Excel Save Error: ", e$message, "\n"), file = error_log_con)
            })
        }
    }
}

close(error_log_con)

cat("\nSimulation 4 complete!\n")
cat(paste("Final results saved to:", rdata_filepath, " and ", excel_filepath, "\n"))
cat(paste("Errors (if any) logged to:", error_log_filepath, "\n"))

final_df_scenario4 <- rbindlist(results_scenario4_list, fill = TRUE)
print("Final Simulation 4 Results:")
print(final_df_scenario4)

#Plot the results
plot_df_scenario4 <- copy(final_df_scenario4) # Create a copy to avoid modifying the original
plot_df_scenario4[, recommendation_numeric := as.numeric(ifelse(recommendation == "***" | recommendation == "ERROR", NA, recommendation))]
plot_df_scenario4[, dependencies_factor := factor(dependencies)]
ggplot(plot_df_scenario4, aes(x = coef_mean, y = recommendation_numeric, color = dependencies_factor))
  geom_point(size = 4, alpha = 0.7) +
  # Add lines to connect points for each level of dependencies
  geom_line(aes(group = dependencies_factor), alpha = 0.5) +
  labs(
    title = paste0("Recommended Sample Size (N_VARS = ", CONSTANT_N_VARS, ", Model = ", CONSTANT_MODEL_TYPE, ")"),
    subtitle = "Varying Signal Strength and Dependencies",
    x = "Signal Strength (Coefficient Mean)",
    y = "Minimum Number of Observations",
    color = "Dependencies" # Legend title
  ) +
  theme_minimal() +
  theme(
    # Legend Text (the individual items/levels in the legend)
    legend.text = element_text(size = 12),

    # Legend Title (e.g., "Dependencies")
    legend.title = element_text(size = 14, face = "bold"),

    # Axis Titles (x and y labels)
    axis.title = element_text(size = 14, face = "bold"),

    # Axis Tick Labels (the numbers or categories on the axes)
    axis.text = element_text(size = 12),

    # Plot Title
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5) # Add subtitle styling
  ) +
  scale_color_hue()





# Example usage of the wrapper function
#NLP scenario
recommendation_NLP <- recommend_sample_size(
    n_obs = 100000, # Population size
    n_vars = 300, # Number of variables in the mega-population
    coef_mean = 0.1, # Mean of the coefficients
    coef_sd = 0.05, # Standard deviation of the coefficients
    dependencies = 0.1, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 150,
    repetitions = 10,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_NLP

#Sociology scenario
recommendation_sociology <- recommend_sample_size(
    n_obs = 100000, # Population size
    n_vars = 7, # Number of variables in the mega-population
    coef_mean = 0.1, # Mean of the coefficients
    coef_sd = 0.6, # Standard deviation of the coefficients
    dependencies = 0.05, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 150,
    repetitions = 10,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_sociology

#Neuroimaging scenario
recommendation_neuroimaging <- recommend_sample_size(
    n_obs = 100000, # Population size
    n_vars = 3000, # Number of variables in the mega-population
    coef_mean = 0.2, # Mean of the coefficients
    coef_sd = 0.1, # Standard deviation of the coefficients
    dependencies = 0.05, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 10000, # Max sample size range (budget constrains)
    step_size = 150,
    repetitions = 10,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)

#Psychology scenario
recommendation_psychology <- recommend_sample_size(
    n_obs = 100000, # Population size
    n_vars = 9, # Number of variables in the mega-population
    coef_mean = 0.3, # Mean of the coefficients
    coef_sd = 0.25, # Standard deviation of the coefficients
    dependencies = 0.2, #correlation between variables
    model_type = "regularized logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 100, #Fixed increment of sample size candidates
    repetitions = 10, # Number of replications for each sample size
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_psychology

recommendation_banai_empiricalCase <- recommend_sample_size(
    n_obs = 1000000, # Population size
    n_vars = 9, # Number of variables in the mega-population
    coef_mean = 0.05, # Mean of the coefficients
    coef_sd = 0.02, # Standard deviation of the coefficients
    dependencies = 0.55, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 50,
    repetitions = 30,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_banai_empiricalCase

recommendation_banai_informed <- recommend_sample_size(
    n_obs = 1000000, # Population size
    n_vars = 9, # Number of variables in the mega-population
    coef_mean = 0.3, # Mean of the coefficients
    coef_sd = 0.02, # Standard deviation of the coefficients
    dependencies = 0.55, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 50,
    repetitions = 30,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_banai_realistic

recommendation_banai_conservative <- recommend_sample_size(
    n_obs = 1000000, # Population size
    n_vars = 9, # Number of variables in the mega-population
    coef_mean = 0.01, # Mean of the coefficients
    coef_sd = 0.10, # Standard deviation of the coefficients
    dependencies = 0.20, #correlation between variables
    model_type = "logistic regression", # Model type to use
    start_size = 100, # Min sample size range
    end_size = 5000, # Max sample size range (budget constrains)
    step_size = 50,
    repetitions = 30,
    convergence_threshold = 0.01, #corresponds to 1% difference in performance
    n_cores = 7 #number of cores to use for parallel processing
)
recommendation_banai_conservative