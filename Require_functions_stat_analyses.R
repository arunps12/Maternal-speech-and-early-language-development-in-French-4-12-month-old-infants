# Functions for stat models, boot, and plotting
# 1. lmer model
lmer.full.reduced.null.compare <- function(fullmformula, 
                                           optimizer = "bobyqa", 
                                           maxfun = 100000,
                                           fullm = TRUE,
                                           redm = FALSE, 
                                           redmformula = NULL, 
                                           nullm = FALSE, 
                                           nullmformula = NULL,
                                           data_list = list(...),
                                           marginal_means = FALSE,
                                           specs = NULL,
                                           ...) {
  
  # Load required packages
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("The 'lme4' package is required but is not installed.")
  }
  if (!requireNamespace("lmerTest", quietly = TRUE)) {
    stop("The 'lmerTest' package is required but is not installed.")
  }
  if (!requireNamespace("car", quietly = TRUE)) {
    stop("The 'car' package is required but is not installed.")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required but is not installed.")
  }
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("The 'dplyr' package is required but is not installed.")
  }
  if (!requireNamespace("emmeans", quietly = TRUE)) {
    stop("The 'emmeans' package is required but is not installed.")
  }
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("The 'crayon' package is required but is not installed.")
  }
  if (!requireNamespace("nortest", quietly = TRUE)) {
    stop("The 'nortest' package is required but is not installed.")
  }
  
  library(lme4)
  library(lmerTest)
  library(car)
  library(ggplot2)
  library(dplyr)
  library(emmeans)
  library(crayon)
  library(nortest)
  # Combine input dataframes
  combined_data <- bind_rows(data_list)
  cat(blue("Structure of combined data:\n"))
  str(combined_data)
  
  # Extract the response variable 
  terms_obj <- terms(fullmformula)
  response_var <- as.character(attr(terms_obj, "variables"))[2]
  cat(blue("Response variable is:"), response_var, "\n")
  
  # Standardize all numeric columns except the response variable
  numeric_columns <- sapply(combined_data, is.numeric)
  numeric_columns <- setdiff(names(combined_data)[which(numeric_columns)], response_var)
  combined_data[numeric_columns] <- lapply(combined_data[numeric_columns], scale)
  # Convert the response variable to numeric if it's not already
  combined_data[[response_var]] <- as.numeric(as.character(combined_data[[response_var]]))
  # Check structure of combined data after scaling
  cat(blue("Structure of combined data after scaling:\n"))
  str(combined_data)
  
  # Histogram of Response variable
  p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
    geom_histogram(bins = 30) +
    labs(title = "Histogram of Response Variable")
  print(p_histogram)  
  
  # Check if the response variable is normally distributed
  if (nrow(combined_data) > 5000) {
    normal_test <- ad.test(combined_data[[response_var]])
  } else {
    normal_test <- shapiro.test(combined_data[[response_var]])
  }
  
  
  # Check normality based on the p-value
  if (normal_test$p.value < 0.05) {
    combined_data[[response_var]] <- log(combined_data[[response_var]] + 1)
    cat(blue("Response variable is not normally distributed. Log transformation applied.\n"))
    cat(blue("Structure of combined data after transformation:\n"))
    str(combined_data)
  
  
    
    # Histogram of Response variable after transformation
    p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
      geom_histogram(bins = 30) +
      labs(title = "Histogram of Log-transformed Response Variable")
    print(p_histogram)  
  } else {
    cat(blue("Response variable is normally distributed; no transformation applied.\n"))
  }
  
  # Initialize list to store emmeans
  #emmeans_list <- list()
  
  # Fitting the full model
  fit_model <- NULL
  if (fullm && !is.null(fullmformula)) {
    cat(blue("Fitting Full Model.\n"))
    fit_model <- tryCatch({
      lmer(fullmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting the full model:", e$message, "\n"))
      return(NULL)
    })
    
    # Check if the full model was fitted successfully
    if (is.null(fit_model)) {
      cat(blue("Full model failed. Consider using a different optimizer or increasing maxfun.\n"))
    } else {
      fit_model_REML <- lmerTest::lmer(fullmformula, data = combined_data, REML = TRUE, 
                                       control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(fit_model)) {
        cat(blue("Full model is singular. Consider simplifying the random effects structure.\n"))
      }
      
      # Residuals and fitted values plots for full model
      residuals_full <- resid(fit_model)
      fitted_values_full <- fitted(fit_model)
      
      p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values_full, residuals = residuals_full), aes(x = fitted, y = residuals)) +
        geom_point() +
        geom_smooth(method = "lm") +
        labs(title = "Residuals vs Fitted Values for Full Model", x = "Fitted Values", y = "Residuals")
      print(p_residuals_vs_fitted)  
      
      p_qq_plot <- ggplot(data = data.frame(residuals = residuals_full), aes(sample = residuals)) +
        geom_qq() +
        geom_qq_line() +
        labs(title = "Q-Q Plot of Residuals for Full Model")
      print(p_qq_plot)  
      
      # Extract model summary for full model
      model_summary <- summary(fit_model_REML)
      cat(blue("Summary of the Full Model:\n"))
      print(model_summary)
      
      logLik_full <- logLik(fit_model)
      df_full <- attr(logLik_full, "df")
      cat(blue("Full Model: Log-Likelihood =", round(as.numeric(logLik_full), 2), 
               ", Degrees of Freedom =", df_full, "\n"))
      vif_full <- vif(fit_model)
      cat(blue("Max VIF value of the Full Model:", max(vif_full), "\n"))
      
      # Compute emmeans for full model if requested
      if (marginal_means) {
        if (is.null(specs)) {
          stop("Please provide 'specs' argument for computing emmeans.")
        }
        cat(blue("Computing estimated marginal means for the Full Model.\n"))
        emm <- emmeans(fit_model, specs = specs)
        print(summary(emm))
        #emmeans_list$full_model_emm <- emm_full
      }
    }
  }
  # Fitting the null model
  null_fit_model <- NULL
  if (nullm && !is.null(nullmformula)) {
    cat(blue("Fitting Null Model.\n"))
    null_fit_model <- tryCatch({
      lmer(nullmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting null model:", e$message, "\n"))
      return(NULL)
    })
  }
  
  # Compare models using ANOVA
  if (nullm && !is.null(nullmformula)) {
    if (fullm && !is.null(fullmformula) && !is.null(fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Full and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      full_null_model_comparison <- anova(fit_model, null_fit_model)
      print(full_null_model_comparison)
    }
  }
  
  # Fitting the reduced model
  red_fit_model <- NULL
  if (redm && !is.null(redmformula)) {
    cat(blue("Fitting Reduced Model.\n"))
    red_fit_model <- tryCatch({
      lmer(redmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting the reduced model:", e$message, "\n"))
      return(NULL)
    })
    
    if (is.null(red_fit_model)) {
      cat(blue("Reduced model failed. Consider using a different optimizer or increasing maxfun.\n"))
    } else {
      red_fit_model_REML <- lmerTest::lmer(redmformula, data = combined_data, REML = TRUE, 
                                           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(red_fit_model)) {
        cat(blue("Reduced model is singular. Consider simplifying the random effects structure.\n"))
      }
      
      # Residuals and fitted values plots for reduced model
      residuals_red <- resid(red_fit_model)
      fitted_values_red <- fitted(red_fit_model)
      
      p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values_red, residuals = residuals_red), aes(x = fitted, y = residuals)) +
        geom_point() +
        geom_smooth(method = "lm") +
        labs(title = "Residuals vs Fitted Values for Reduced Model", x = "Fitted Values", y = "Residuals")
      print(p_residuals_vs_fitted)  
      
      p_qq_plot <- ggplot(data = data.frame(residuals = residuals_red), aes(sample = residuals)) +
        geom_qq() +
        geom_qq_line() +
        labs(title = "Q-Q Plot of Residuals for Reduced Model")
      print(p_qq_plot)  
      
      # Extract model summary for reduced model
      red_model_summary <- summary(red_fit_model_REML)
      cat(blue("Summary of the Reduced Model:\n"))
      print(red_model_summary)
      
      logLik_reduced <- logLik(red_fit_model)
      df_reduced <- attr(logLik_reduced, "df") 
      cat(blue("Reduced Model: Log-Likelihood =", round(as.numeric(logLik_reduced), 2), 
               ", Degrees of Freedom =", df_reduced, "\n"))
      vif_red <- vif(red_fit_model)
      cat(blue("Max VIF value of the Reduced Model:", max(vif_red), "\n"))
      
      # Compute emmeans for reduced model if requested
      if (marginal_means) {
        if (is.null(specs)) {
          stop("Please provide 'specs' argument for computing emmeans.")
        }
        cat(blue("Computing estimated marginal means for the Reduced Model.\n"))
        emm <- emmeans(red_fit_model, specs = specs)
        print(summary(emm))
        #emmeans_list$reduced_model_emm <- emm_red
      }
    }
  }
  
  # Fitting the null model
  null_fit_model <- NULL
  if (nullm && !is.null(nullmformula)) {
    cat(blue("Fitting Null Model.\n"))
    null_fit_model <- tryCatch({
      lmer(nullmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting null model:", e$message, "\n"))
      return(NULL)
    })
  }
  
  # Compare models using ANOVA
  if (nullm && !is.null(nullmformula)) {
    if (redm && !is.null(redmformula) && !is.null(red_fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Reduced and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      reduced_null_model_comparison <- anova(red_fit_model, null_fit_model)
      print(reduced_null_model_comparison)
    } 
  }
  
  # Return the results
  cat(blue("Returning the fitted models, data, optimizer settings, and emmeans (if computed).\n"))
  return(list(full_model = fit_model, 
              reduced_model = red_fit_model, 
              null_model = null_fit_model, 
              fit_data = combined_data, 
              optimizer = optimizer, 
              maxfun = maxfun, 
              emmeans = emm))
}





# glmmTB model
glmmTB.full.reduced.null.compare <- function(fullmformula,
                                             family,
                                             zi_formula = NULL,
                                             fullm = FALSE,
                                             redm = FALSE, 
                                             redmformula = NULL, 
                                             nullm = FALSE, 
                                             nullmformula = NULL,
                                             data_list = list(...),
                                             marginal_means = FALSE,
                                             specs = NULL,
                                             ...) {
  # Load required packages
  if (!requireNamespace("glmmTMB", quietly = TRUE)) {
    stop("The 'glmmTMB' package is required but is not installed.")
  }
  if (!requireNamespace("emmeans", quietly = TRUE)) {
    stop("The 'emmeans' package is required but is not installed.")
  }
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("The 'crayon' package is required but is not installed.")
  }
  if (!requireNamespace("DHARMa", quietly = TRUE)) {
    stop("The 'DHARMa' package is required but is not installed.")
  }
  if (!requireNamespace("performance", quietly = TRUE)) {
    stop("The 'performance' package is required but is not installed.")
  }
  
  library(glmmTMB)
  library(emmeans)
  library(crayon)
  library(DHARMa)
  library(performance)
  
  # Combine input dataframes
  combined_data <- bind_rows(data_list)
  cat(blue("Structure of combined data:\n"))
  str(combined_data)
  
  # Extract the response variable 
  terms_obj <- terms(fullmformula)
  response_var <- as.character(attr(terms_obj, "variables"))[2]  # Extract the response variable
  print(response_var)
  
  # Standardize all numeric columns except the response variable
  numeric_columns <- sapply(combined_data, is.numeric)
  numeric_columns <- setdiff(names(combined_data)[which(numeric_columns)], response_var)
  combined_data[numeric_columns] <- lapply(combined_data[numeric_columns], scale)  # Standardize numeric columns
  # Convert the response variable to numeric if it's not already
  combined_data[[response_var]] <- as.numeric(as.character(combined_data[[response_var]]))
  # Check structure of combined data after scaling
  cat(blue("Structure of combined data after scaling:\n"))
  str(combined_data)
  
  epsilon <- 1e-6
  combined_data[[response_var]] <- ifelse(combined_data[[response_var]] == 0, epsilon,
                                          ifelse(combined_data[[response_var]] == 1, 1 - epsilon, combined_data[[response_var]]))
  
  # Histogram of Response variable
  p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
    geom_histogram(bins = 30) +
    labs(title = "Histogram of Response Variable")
  print(p_histogram)
  
  # Initialize emmeans variable to NULL
  #emm <- NULL
  # Fit the full model using glmmTMB
  fit_model <- NULL
  if (fullm && !is.null(fullmformula)) {
    cat(blue("Fitting Full Model.\n"))
    fit_model <- tryCatch({
      glmmTMB(formula = fullmformula, data = combined_data, family = family, ziformula = zi_formula) 
    }, error = function(e) {
      cat(blue("Error in fitting full model:", e$message, "\n"))
      return(NULL)
    })
    full_model_summary <- summary(fit_model)
    cat(blue("Summary of the full model:\n"))
    print(full_model_summary)
    cat(blue("Simulate residuals for the fitted full glmmTMB model.\n"))
    simulated_residuals_full <- simulateResiduals(fittedModel = fit_model, plot = T)
    
    cat(blue("Check for overdispersion with DHARMa test for full model.\n"))
    dispersion_test_full <- DHARMa::testDispersion(simulationOutput = simulated_residuals_full)
    
    # Print the results of the dispersion test
    cat("DHARMa dispersion for full model test results:\n")
    cat("Dispersion:", dispersion_test_full$statistic, "\n")
    cat("p-value:", dispersion_test_full$p.value, "\n")
    cat("Alternative hypothesis:", dispersion_test_full$alternative, "\n")
    
    # Compute VIF (multicollinearity check)
    collinearity_results_full <- check_collinearity(fit_model)
    cat(blue("VIF for the full model:\n"))
    print(collinearity_results_full)
    
    logLik_full <- logLik(fit_model)
    df_full <- attr(logLik_full, "df")
    cat(blue("Full Model: Log-Likelihood =", round(as.numeric(logLik_full), 2), ", Degrees of Freedom =", df_full, "\n"))
    
    # Compute marginal means if requested
    if (marginal_means) {
      if (is.null(specs)) {
        stop("Please provide 'specs' argument for computing emmeans.")
      }
      cat(blue("Computing estimated marginal means for the Full Model.\n"))
      emm <- emmeans(fit_model, specs = specs, type = "response")
      print(summary(emm))
    }
  }
  
  #if (fullm && !is.null(fullmformula)) {
  
  #}
  
  # Fit the reduced model if specified
  red_fit_model <- NULL
  if (redm && !is.null(redmformula)) {
    cat(blue("Fitting Reduced Model.\n"))
    red_fit_model <- tryCatch({
      glmmTMB(formula = redmformula, data = combined_data, family = family, ziformula = zi_formula)
    }, error = function(e) {
      cat(blue("Error in fitting reduced model:", e$message, "\n"))
      return(NULL)
    })
    red_model_summary <- summary(red_fit_model)
    cat(blue("Summary of the reduced model:\n"))
    print(red_model_summary)
    cat(blue("Simulate residuals for the fitted reduced glmmTMB model.\n"))
    simulated_residuals_red <- simulateResiduals(fittedModel = red_fit_model, plot = T)
    
    cat(blue("Check for overdispersion with DHARMa test for reduced model.\n"))
    dispersion_test_red <- DHARMa::testDispersion(simulationOutput = simulated_residuals_red)
    
    # Print the results of the dispersion test
    cat("DHARMa dispersion for reduced model test results:\n")
    cat("Dispersion:", dispersion_test_red$statistic, "\n")
    cat("p-value:", dispersion_test_red$p.value, "\n")
    cat("Alternative hypothesis:", dispersion_test_red$alternative, "\n")
    
    # Compute VIF (multicollinearity check)
    collinearity_results_red <- check_collinearity(red_fit_model)
    cat(blue("VIF for the reduced model:\n"))
    print(collinearity_results_red)
    
    logLik_red <- logLik(red_fit_model)
    df_red <- attr(logLik_red, "df")
    cat(blue("Reduced Model: Log-Likelihood =", round(as.numeric(logLik_red), 2), ", Degrees of Freedom =", df_red, "\n"))
    
    # Compute marginal means if requested
    if (marginal_means) {
      if (is.null(specs)) {
        stop("Please provide 'specs' argument for computing emmeans.")
      }
      cat(blue("Computing estimated marginal means for the Reduced Model.\n"))
      emm <- emmeans(red_fit_model, specs = specs, type = "response")
      print(summary(emm))
    }
  }
  
  #if (redm && !is.null(redmformula)) {
  
  #}
  
  # Fit the null model if specified
  null_fit_model <- NULL
  if (nullm && !is.null(nullmformula)) {
    cat(blue("Fitting Null Model.\n"))
    null_fit_model <- tryCatch({
      glmmTMB(formula = nullmformula, data = combined_data, family = family, ziformula = zi_formula)
    }, error = function(e) {
      cat(blue("Error in fitting null model:", e$message, "\n"))
      return(NULL)
    })
  }
  
  # Compare models using ANOVA
  if (nullm && !is.null(nullmformula)) {
    if (redm && !is.null(redmformula) && !is.null(red_fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Reduced and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      reduced_null_model_comparison <- anova(red_fit_model, null_fit_model)
      print(reduced_null_model_comparison)
    } else if (fullm && !is.null(fullmformula) && !is.null(fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Full and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      full_null_model_comparison <- anova(fit_model, null_fit_model)
      print(full_null_model_comparison)
    }
  }
  
  # Return the fitted models
  return(list(full_model = fit_model, reduced_model = red_fit_model, null_model = null_fit_model, fit_data = combined_data, emmeans = emm))
}


# 3. fun for bootstrapped predictions
boot.ci.predict.lmer <- function(m, optimizer, maxfun, data, pred.data, reqcol, centercol = NULL, nboots = NULL, link = "identity", keep.boots = FALSE) {
  
  
  # Load required packages
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("The 'lme4' package is required but is not installed.")
  }
  if (!requireNamespace("boot", quietly = TRUE)) {
    stop("The 'boot' package is required but is not installed.")
  }
  
  # Load the libraries
  library(lme4)
  library(boot)
  # Initialize a counter singular fit
  #warning_counter <- 0
  singular_fit_counter <- 0
  
  # Filter pred.data to use only variables specified in 'reqcol'
  pred.data <- pred.data[ , reqcol, drop = FALSE]
  #print(pred.data)
  
  # Centering variables
  if (!is.null(centercol)) {
    for (var in centercol) {
      if (is.numeric(data[[var]])) {
        # Center numeric variables
        mean_value <- mean(data[[var]], na.rm = TRUE)
        pred.data[[var]] <- rep(mean_value, nrow(pred.data))  # Use mean value for predictions
      } else if (is.character(data[[var]])) {
        # Use the most frequent level for categorical variables
        data[[var]] = as.factor(data[[var]])
        most_frequent_level <- names(which.max(table(data[[var]])))
        pred.data[[var]] <- factor(rep(most_frequent_level, nrow(pred.data)), levels = levels(data[[var]]))
      }
    }
  }
  #print(pred.data)
  # Predict values based on the model
  prediction <- predict(m, newdata = pred.data, type = "response", re.form = ~0)
  #print(prediction)
  
  # Perform bootstrapping to calculate confidence intervals
  if (!is.null(nboots)) {
    set.seed(123)
    boots_results <- boot(data, statistic = function(data, indices) {
      boot_model <- withCallingHandlers(
        {
          update(m, formula = formula(m), data = data[indices, ], REML = FALSE, control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
        },
        warning = function(w) {
          if (grepl("convergence|eigenvalue", conditionMessage(w))) {  # Filter specific warnings
            #warning_counter <<- warning_counter + 1
          }
          invokeRestart("muffleWarning")
        }
      )
      
      #boot_model <- update(m, formula = formula(m), data = data[indices, ], REML = FALSE, control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(boot_model)) {
        cat(blue("singular fit \n"))
        singular_fit_counter <<- singular_fit_counter + 1
        return(rep(NA, nrow(pred.data)))  # Return NA if the model is singular
      }
      
      #Predict using the bootstrapped model
      predictions <- predict(boot_model, newdata = pred.data, type = "response", re.form = ~0)  
      
      if (link == "log") {
        boot_preds <- exp(predictions)
      } else if (link == "inverse") {
        boot_preds <- 1 / predictions
      } else if (link == "identity") {
        boot_preds <- predictions
      } else {
        stop("Unsupported link function. Use 'log', 'inverse', or 'identity'.")
      }
      return(boot_preds)
    }, R = nboots)
    
    # Calculate confidence intervals
    # Filter out NA predictions (from singular fits)
    valid_preds <- boots_results$t[complete.cases(boots_results$t), ]
    sim_preds <- t(valid_preds)
    lower_ci <- apply(sim_preds, 1, quantile, probs = 0.025) 
    upper_ci <- apply(sim_preds, 1, quantile, probs = 0.975) 
    #print(lower_ci)
    #print(upper_ci)
    
    # Print the number of warnings that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    # Print the number of warnings and singular fits that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    cat(blue("Number of singular fits during bootstrapping:", singular_fit_counter, "out of", nboots, "bootstrap iterations.\n"))
    
    # If keep.boots = TRUE, return the bootstrapped predictions along with the summary results
    if (keep.boots) {
      return(list(predictions = data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci),
                  boot_samples = boots_results$t))
    }
  } else {
    lower_ci <- upper_ci <- NULL
  }
  
  
  # If keep.boots = FALSE, return only the summary predictions
  return(data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci))
}


# 4. Function for bootstrappin glmmtmb
boot.ci.predict.glmmTMB <- function(m, data, pred.data, reqcol, centercol = NULL, nboots = NULL, keep.boots = FALSE) {
  
  # Load required libraries
  if (!requireNamespace("glmmTMB", quietly = TRUE)) {
    stop("The 'glmmTMB' package is required but is not installed.")
  }
  if (!requireNamespace("boot", quietly = TRUE)) {
    stop("The 'boot' package is required but is not installed.")
  }
  
  # Load the libraries
  library(glmmTMB)
  library(boot)
  # Initialize a counter singular fit
  #warning_counter <- 0
  singular_fit_counter <- 0
  
  # Filter pred.data to use only variables specified in 'reqcol'
  pred.data <- pred.data[ , reqcol, drop = FALSE]
  #print(pred.data)
  
  # Centering variables
  if (!is.null(centercol)) {
    for (var in centercol) {
      if (is.numeric(data[[var]])) {
        # Center numeric variables
        mean_value <- mean(data[[var]], na.rm = TRUE)
        pred.data[[var]] <- rep(mean_value, nrow(pred.data))  # Use mean value for predictions
      } else if (is.character(data[[var]])) {
        # Use the most frequent level for categorical variables
        data[[var]] = as.factor(data[[var]])
        most_frequent_level <- names(which.max(table(data[[var]])))
        pred.data[[var]] <- factor(rep(most_frequent_level, nrow(pred.data)), levels = levels(data[[var]]))
      }
    }
  }
  
  # Predict values based on the model
  prediction <- predict(m, newdata = pred.data, type = "response", re.form = ~0)
  #print(prediction)
  
  # Perform bootstrapping to calculate confidence intervals
  if (!is.null(nboots)) {
    set.seed(123)
    boots_results <- boot(data, statistic = function(data, indices) {
      boot_model <- withCallingHandlers(
        {
          update(m, formula = formula(m), data = data[indices, ], family = family(m))
        },
        warning = function(w) {
          if (grepl("convergence|eigenvalue", conditionMessage(w))) {  # Filter specific warnings
            #warning_counter <<- warning_counter + 1
          }
          invokeRestart("muffleWarning")
        }
      )
      
      #Predict using the bootstrapped model
      
      predictions <- predict(boot_model, newdata = pred.data, type = "response", re.form = ~0)  
      
      return(predictions)
    }, R = nboots)
    
    # Calculate confidence intervals
    # Filter out NA predictions (from singular fits)
    valid_preds <- boots_results$t[complete.cases(boots_results$t), ]
    sim_preds <- t(valid_preds)
    lower_ci <- apply(sim_preds, 1, quantile, probs = 0.025) 
    upper_ci <- apply(sim_preds, 1, quantile, probs = 0.975) 
    #print(lower_ci)
    #print(upper_ci)
    
    # Print the number of warnings that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    # Print the number of warnings and singular fits that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    cat(blue("Number of singular fits during bootstrapping:", singular_fit_counter, "out of", nboots, "bootstrap iterations.\n"))
    
    # If keep.boots = TRUE, return the bootstrapped predictions along with the summary results
    if (keep.boots) {
      return(list(predictions = data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci),
                  boot_samples = boots_results$t))
    }
  } else {
    lower_ci <- upper_ci <- NULL
  }
  
  
  # If keep.boots = FALSE, return only the summary predictions
  return(data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci))
}

# Function to plot fit values as a function of AgeInDays with confidence intervals
continuous_fit_ci_plot <- function(plot.data.list, 
                                   coefs.list, 
                                   plot.data.col.x,
                                   plot.data.col.y,
                                   x.labs, 
                                   y.labs,
                                   x.lim = NULL,
                                   y.lim = NULL,
                                   div = NULL,
                                   x.ax.transformation = FALSE,
                                   inv_x_transform_fun.list = NULL,  # List of inverse x transformations
                                   y.ax.transformation = FALSE,
                                   inv_y_transform_fun.list = NULL,  # List of inverse y transformations
                                   x.breaks = NULL,
                                   y.breaks = NULL,
                                   y_grid_log = FALSE,
                                   legend.labels = NULL,
                                   colors = c("#FCA636", "#6A00A8"), # Colors for points and lines
                                   alpha_level = 0.1,               # Transparency level for points
                                   ribbon_colors = c("grey", "grey") # Colors for confidence interval ribbons
) {
  # Load required libraries
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required but is not installed.")
  }
  library(ggplot2)
  
  # Create the base plot
  p <- ggplot()
  
  # Loop through each dataset in plot.data.list and coefs.list
  for (i in seq_along(plot.data.list)) {
    plot.data <- plot.data.list[[i]]
    coefs <- coefs.list[[i]]
    
    # Apply inverse transformations if specified
    if (x.ax.transformation && !is.null(inv_x_transform_fun.list)) {
      plot.data[[plot.data.col.x]] <- inv_x_transform_fun.list[[i]](plot.data[[plot.data.col.x]])
      coefs[[plot.data.col.x]] <- inv_x_transform_fun.list[[i]](coefs[[plot.data.col.x]])
    }
    if (y.ax.transformation && !is.null(inv_y_transform_fun.list)) {
      plot.data[[plot.data.col.y]] <- inv_y_transform_fun.list[[i]](plot.data[[plot.data.col.y]])
      coefs$fit <- inv_y_transform_fun.list[[i]](coefs$fit)
      coefs$lwr <- inv_y_transform_fun.list[[i]](coefs$lwr)
      coefs$upr <- inv_y_transform_fun.list[[i]](coefs$upr)
    }
    
    # Scale y values if needed
    if (!is.null(div)) {
      plot.data[[plot.data.col.y]] <- plot.data[[plot.data.col.y]] / div
      coefs$fit <- coefs$fit / div
      coefs$lwr <- coefs$lwr / div
      coefs$upr <- coefs$upr / div
    }
    
    # Add original data points for each dataset
    p <- p +
      geom_point(data = plot.data, aes_string(x = plot.data.col.x, y = plot.data.col.y, color = as.factor(i)),
                 alpha = alpha_level, size = 3, show.legend = TRUE)
    
    # Add fitted line for each dataset
    p <- p +
      geom_line(data = coefs, aes_string(x = plot.data.col.x, y = "fit", color = as.factor(i)), 
                size = 1, show.legend = TRUE)
    
    # Add confidence interval ribbon for each dataset
    p <- p +
      geom_ribbon(data = coefs, aes_string(x = plot.data.col.x, ymin = "lwr", ymax = "upr", fill = as.factor(i)),
                  alpha = 0.5, show.legend = FALSE)  # Semi-transparent ribbon
  }
  
  # Labels, limits, and breaks
  p <- p + labs(x = x.labs, y = y.labs, color = "", fill = "") +
    #scale_x_continuous(limits = x.lim, breaks = x.breaks) +
    #scale_y_continuous(limits = y.lim, breaks = y.breaks) +
    scale_color_manual(values = colors, labels = legend.labels) +
    scale_fill_manual(values = ribbon_colors, labels = legend.labels) +  # Custom ribbon colors
    theme_minimal() +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      legend.position = "top"  # Adjust legend position to top
    )
  if (y_grid_log){
    p <- p + scale_y_log10()
  }
  
  return(p)
}



