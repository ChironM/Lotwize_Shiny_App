# Load required packages
library(tidyverse)
library(caret)
library(corrplot)
library(xgboost)
library(recipes)
library(scales)
library(shiny)
library(RANN)

# Read data
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
data <- read.csv("lotwize_case.csv")
crime_data <- read.csv("hci_crime_752_pl_co_re_ca_2000-2013_21oct15-ada.csv")

# Select important features
selected_features <- c(
  "price", "latitude", "longitude", "schools.0.distance", # fix column names with slashes
  "livingAreaValue", "lotSize", "bedrooms", "bathrooms",
  "homeType", "yearBuilt", "priceHistory.0.price",
  "taxAssessedValue", "zestimate", "lastSoldPrice",
  "nearbyHomes.0.price", "nearbyHomes.0.lotSize",
  "nearbyHomes.0.livingArea", "monthlyHoaFee",
  "taxHistory.0.value", "mortgageRates.thirtyYearFixedRate",
  "photoCount", "zipcode", "county"
)


data <- data[selected_features]

# Identifying continuous and categorical features
continuous_features <- c(
  'price', 'latitude', 'longitude', 'schools.0.distance', 'livingAreaValue', 'lotSize', 
  'priceHistory.0.price', 'taxAssessedValue', 'zestimate', 'lastSoldPrice', 'nearbyHomes.0.price', 
  'nearbyHomes.0.lotSize', 'monthlyHoaFee', 'taxHistory.0.value', 'mortgageRates.thirtyYearFixedRate', 
  'photoCount', 'yearBuilt', 'bedrooms', 'bathrooms'
)

categorical_features <- c('homeType', 'zipcode')

# Dropping variables to fix multicollinearity
data <- data %>%
  select(-c('taxHistory.0.value', 'lastSoldPrice', 'priceHistory.0.price', 
            'mortgageRates.thirtyYearFixedRate', 'zestimate', 
            'nearbyHomes.0.price', 'taxAssessedValue', 'nearbyHomes.0.livingArea'))

continuous_features <- c(
  'price', 'latitude', 'longitude', 'schools.0.distance', 'livingAreaValue', 'lotSize', 
  'nearbyHomes.0.lotSize', 'monthlyHoaFee',
  'photoCount', 'yearBuilt', 'bedrooms', 'bathrooms'
)

# Merging datasets
crime_data$county <- paste0(crime_data$county, ' County')
crime_data <- crime_data %>% drop_na(county)
data <- merge(data, crime_data[, c('county', 'rate')], by='county', all.x=TRUE)

# Dropping county column
data <- data %>%
  select(-county)

# Cleaning for outliers
z_threshold <- 2
z <- abs(scale(data[continuous_features]))
outlier_indices <- which(z > z_threshold)
data <- data[-outlier_indices, ]

# Check for missing values in categorical features
colSums(is.na(data[categorical_features]))

# Converting categorical features to strings
data[categorical_features] <- lapply(data[categorical_features], as.character)

# Creating dummy variables
cat_dummies <- model.matrix(~ . - 1, data = data[categorical_features])
cat_dummies <- as.data.frame(cat_dummies)

# Ensure the number of rows match before binding
cat_dummies <- cat_dummies[1:nrow(data), ]

# Dropping original categorical variables from dataset
levels_data <- data
data <- data %>%
  select(-c(homeType, zipcode)) %>%
  bind_cols(cat_dummies)

# Dropping specified columns
data <- data %>%
  select(-c(latitude, longitude, schools.0.distance, livingAreaValue, nearbyHomes.0.lotSize, monthlyHoaFee, photoCount, rate))

# Scaling data
scaler <- preProcess(data, method = 'range')
data <- predict(scaler, data)

# Dropping specific columns
data <- data %>%
  select(-homeTypeHOME_TYPE_UNKNOWN)

# Imputing missing values
preImpute <- preProcess(data, method = 'knnImpute')
data <- predict(preImpute, data)

# Split data
set.seed(42)
train_index <- createDataPartition(data$price, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Prepare matrices for XGBoost
X_train <- as.matrix(train_data %>% select(-price))
y_train <- train_data$price
X_test <- as.matrix(test_data %>% select(-price))
y_test <- test_data$price

# Set parameters for XGBoost
params <- list(
  objective = "reg:squarederror",  # Regression task (for continuous target variable)
  booster = "gbtree",              # Using tree boosting
  eta = 0.1,                       # Learning rate
  max_depth = 6,                   # Maximum depth of trees
  subsample = 0.8,                 # Subsample ratio of the training set
  colsample_bytree = 0.8           # Subsample ratio of columns when constructing each tree
)

# Train the XGBoost model
model <- xgboost(
  data = X_train,                 # Feature matrix for training
  label = y_train,                # Target variable for training
  params = params,                # Parameters for the model
  nrounds = 100,                  # Number of boosting rounds (iterations)
  verbose = 1                     # Show progress
)

# Calculate and print model metrics
train_pred <- predict(model, X_train)
test_pred <- predict(model, X_test)

train_metrics <- data.frame(
  MSE = mean((y_train - train_pred)^2),
  MAE = mean(abs(y_train - train_pred)),
  MAPE = mean(abs((y_train - train_pred)/y_train)) * 100,
  R2 = cor(y_train, train_pred)^2
)

test_metrics <- data.frame(
  MSE = mean((y_test - test_pred)^2),
  MAE = mean(abs(y_test - test_pred)),
  MAPE = mean(abs((y_test - test_pred)/y_test)) * 100,
  R2 = cor(y_test, test_pred)^2
)

# Calculate feature importance
importance_matrix <- xgb.importance(model = model)

zip_choices <- c("90046", "90066", "90254", "90275", "90277", "90278", "90291",
                 "90712", "90713", "90720", "90740", "90755", "90802", "90803",
                 "90804", "90805", "90806", "90807", "90808", "90815", "91011",
                 "91205", "91302", "91304", "91321", "91350", "91355", "91367",
                 "91384", "91405", "91436", "91722", "91723", "91724", "91754",
                 "91761", "91762", "91764", "91765", "91775", "92009", "92011",
                 "92024", "92037", "92064", "92075", "92101", "92127", "92128",
                 "92129", "92130", "92131", "92335", "92336", "92337", "92501",
                 "92663", "92677", "92801", "92802", "92804", "92805", "92806",
                 "92807", "92808", "92821", "92831", "92833", "92835", "92840",
                 "92843", "92844", "92845", "92860", "92865", "92867", "92870",
                 "92886", "93001", "93003", "93013", "93030", "93033", "93035",
                 "93060", "93101", "93105", "93110", "93111", "93117", "93263",
                 "93301", "93304", "93306", "93307", "93308", "93309", "93311",
                 "93312", "93313", "93314", "93561", "93611", "93612", "93614",
                 "93657", "93703", "93704", "93710", "93711", "93720", "93722",
                 "93726", "93727", "93901", "93905", "93906", "93907", "93908",
                 "93933", "93940", "93950", "93955", "94014", "94015", "94061",
                 "94062", "94063", "94065", "94066", "94080", "94103", "94107",
                 "94110", "94112", "94114", "94116", "94117", "94122", "94124",
                 "94127", "94131", "94132", "94134", "94401", "94402", "94403",
                 "94404", "94506", "94536", "94538", "94539", "94541", "94542",
                 "94544", "94545", "94555", "94563", "95003", "95010", "95014",
                 "95020", "95037", "95050", "95054", "95060", "95062", "95073",
                 "95120", "95746")

# Define UI
ui <- fluidPage(
  titlePanel("House Price Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("zip_code", "Enter ZIP Code:",
                  choices = zip_choices,
                  selected = "zipcode90046"),
      
      numericInput("bedrooms",
                   "Number of Bedrooms:",
                   value = 3,
                   min = 1,
                   max = 10),
      
      numericInput("bathrooms",
                   "Number of Bathrooms:",
                   value = 2,
                   min = 1,
                   max = 10),
      
      numericInput("year_built",
                   "Year Built:",
                   value = 2000,
                   min = 1800,
                   max = 2024),
      
      selectInput("home_type",
                  "Home Type:",
                  choices = c("CONDO", "LOT", "MANUFACTURED", 
                            "MULTI_FAMILY", "SINGLE_FAMILY", "TOWNHOUSE"),
                  selected = "SINGLE_FAMILY"),
      
      numericInput("lot_size",
                   "Lot Size (in sq ft):",
                   value = 5000,
                   min = 0),
      
      actionButton("predict", "Predict Price")
    ),
    
    mainPanel(
      h3("Predicted House Price:"),
      textOutput("prediction_output"),
      
      helpText("This model uses XGBoost to predict house prices based on historical data.
               The prediction is based on the features you input and local market conditions.")
    )
  )
)

# Define recipe for preprocessing
prep_recipe <- recipe(price ~ ., data = train_data) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  step_impute_knn(all_numeric(), -all_outcomes()) %>%
  prep(training = train_data)

# First, save the column names and scaling parameters before the server function
model_columns <- colnames(X_train)
numeric_columns <- c('bedrooms', 'bathrooms', 'yearBuilt', 'lotSize')

 
# Define server
server <- function(input, output, session) {
  prediction <- eventReactive(input$predict, {
    # Create initial data frame with numeric values
    new_data <- data.frame(
      bedrooms = as.numeric(input$bedrooms),
      bathrooms = as.numeric(input$bathrooms),
      yearBuilt = as.numeric(input$year_built),
      lotSize = as.numeric(input$lot_size)
    )
    
    # Create dummy variables manually
    # For zipcode
    for(zip in zip_choices) {
      col_name <- paste0("zipcode", zip)
      new_data[[col_name]] <- as.numeric(input$zip_code == paste0("zipcode", zip))
    }
    
    # For homeType
    home_types <- c("APARTMENT", "CONDO", "LOT", "MANUFACTURED", 
                    "MULTI_FAMILY", "SINGLE_FAMILY", "TOWNHOUSE")
    for(type in home_types) {
      col_name <- paste0("homeType", type)
      new_data[[col_name]] <- as.numeric(input$home_type == type)
    }
    
    # Scale the numeric columns using the original scaling parameters
    for(col in numeric_columns) {
      new_data[[col]] <- (new_data[[col]] - min(train_data[[col]])) / 
        (max(train_data[[col]]) - min(train_data[[col]]))
    }
    
    # Ensure all columns from the training data are present and in the right order
    missing_cols <- setdiff(model_columns, colnames(new_data))
    for(col in missing_cols) {
      new_data[[col]] <- 0
    }
    
    # Reorder columns to match training data exactly
    new_data <- new_data[, model_columns]
    
    # Make prediction
    prediction <- predict(model, as.matrix(new_data))
    
    # Unscale the prediction using the original price range
    price_range <- max(train_data$price) - min(train_data$price)
    price_min <- min(train_data$price)
    unscaled_prediction <- prediction * price_range + price_min
    
    return(unscaled_prediction)
  })
  
  # Render the prediction
  output$prediction_output <- renderText({
    tryCatch({
      req(prediction())
      formatted_price <- dollar_format()(prediction())
      paste("Predicted Price:", formatted_price)
    })
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)