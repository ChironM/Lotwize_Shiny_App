# Load required packages
library(dplyr)
library(readr) 
library(tidyr)
library(purrr)
library(tibble)
library(caret)
library(shiny)
library(scales)
library(leaflet)
library(tidygeocoder)
library(bslib)
library(shinyWidgets)
library(randomForest)

# Read data
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
data <- read.csv("sampled_data.csv")

# Read model
rf_model <- readRDS("rf_model.rds")

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
                 "95120", "95746", "95204", "95207", "95209", "95210", "95212", 
                 "95219", "95242", "95336", "95376", "95746")

home_type_choices <- c("CONDO", "LOT", "MANUFACTURED", 
                       "MULTI_FAMILY", "SINGLE_FAMILY", "TOWNHOUSE")

# Custom CSS
custom_css <- "
.card {
  background-color: #2b2b2b;
  border: 1px solid #3d3d3d;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.form-control {
  background-color: #333333 !important;
  border: 1px solid #444444 !important;
  color: #ffffff !important;
  border-radius: 6px !important;
}

.form-control:focus {
  border-color: #5c7cfa !important;
  box-shadow: 0 0 0 0.2rem rgba(92, 124, 250, 0.25) !important;
}

.btn-primary {
  background-color: #5c7cfa !important;
  border-color: #5c7cfa !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  font-weight: 500 !important;
  transition: all 0.3s ease !important;
}

.btn-primary:hover {
  background-color: #4c6ef5 !important;
  border-color: #4c6ef5 !important;
  transform: translateY(-1px);
}

.selectize-input {
  background-color: #333333 !important;
  border: 1px solid #444444 !important;
  color: #ffffff !important;
}

.selectize-dropdown {
  background-color: #333333 !important;
  border: 1px solid #444444 !important;
  color: #ffffff !important;
}

.selectize-dropdown-content .option {
  color: #ffffff !important;
}

.selectize-dropdown-content .option.active {
  background-color: #5c7cfa !important;
}

#prediction_output {
  font-size: 24px;
  font-weight: 500;
  color: #5c7cfa;
  padding: 15px;
  background-color: #2b2b2b;
  border-radius: 8px;
  display: inline-block;
}

.leaflet-container {
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.section-title {
  color: #ffffff;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #5c7cfa;
}
"

# Define UI
ui <- fluidPage(
  theme = bs_theme(
    version = 5,
    bg = "#1a1a1a",
    fg = "#ffffff",
    primary = "#5c7cfa",
    secondary = "#4c6ef5",
    success = "#51cf66",
    info = "#339af0",
    warning = "#fcc419",
    danger = "#ff6b6b",
    base_font = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    heading_font = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    font_scale = 0.95
  ),
  
  tags$head(
    tags$style(custom_css),
    tags$link(rel = "stylesheet", href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap")
  ),
  
  # Title Panel with modern styling
  div(class = "container-fluid py-4",
      h1("House Price Prediction", 
         style = "color: #ffffff; font-weight: 600; margin-bottom: 30px;",
         class = "text-center")
  ),
  
  # Main Layout
  div(class = "container-fluid",
      fluidRow(
        # Sidebar
        column(4,
               div(class = "card",
                   h3("Input Parameters", class = "section-title"),
                   textInput("address", "Enter Address:", 
                           value = ""),
                   selectInput("zip_code", "Enter ZIP Code:", 
                             choices = zip_choices,
                             selected = "90046"),
                   numericInput("bedrooms", "Number of Bedrooms:", 
                              value = 3, min = 1, max = 10),
                   numericInput("bathrooms", "Number of Bathrooms:", 
                              value = 2, min = 1, max = 10),
                   numericInput("year_built", "Year Built:", 
                              value = 2000, min = 1800, max = 2024),
                   selectInput("home_type", "Home Type:", 
                             choices = home_type_choices,
                             selected = "SINGLE_FAMILY"),
                   numericInput("lot_size", "Lot Size (in sq ft):", 
                              value = 5000, min = 0),
                   numericInput("rate", "Crime Rate:", 
                              value = 5, min = 0),
                   div(style = "margin-top: 25px;",
                       actionButton("predict", "Predict Price", 
                                  class = "btn-primary btn-lg w-100"))
               )
        ),
        
        # Main Panel
        column(8,
               div(class = "card",
                   h3("Predicted House Price", class = "section-title"),
                   textOutput("prediction_output")
               ),
               div(class = "card",
                   h3("Property Location", class = "section-title"),
                   leafletOutput("map", height = 400)
               )
        )
      )
  )
)

server <- function(input, output, session) {
  # Reactive value for storing geocoded coordinates
  coords <- reactiveVal(NULL)
  
  # Observe address changes and geocode
  observe({
    req(input$address)
    
    # Geocode the address
    if (nchar(input$address) > 0) {
      tryCatch({
        # Combine address with ZIP code for better accuracy
        full_address <- paste(input$address, input$zip_code)
        
        # Geocode the address using tidygeocoder
        geo_result <- geo(address = full_address, method = "osm")
        
        if (!is.na(geo_result$lat) && !is.na(geo_result$long)) {
          coords(list(lat = geo_result$lat, lng = geo_result$long))
        }
      }, error = function(e) {
        # Handle geocoding errors silently
        coords(NULL)
      })
    }
  })
  
  # Create reactive prediction function
  prediction <- eventReactive(input$predict, {
    # Create new data frame for prediction
    new_data <- data.frame(
      lotSize = input$lot_size,
      bedrooms = input$bedrooms,
      bathrooms = input$bathrooms,
      yearBuilt = input$year_built,
      rate = input$rate
    )
    
    # Add home type dummy variables
    for (type in home_type_choices) {
      col_name <- paste0("homeType_", type)
      new_data[[col_name]] <- ifelse(input$home_type == type, 1, 0)
    }
    
    # Add zipcode dummy variables
    for (zip in zip_choices) {
      col_name <- paste0("zipcode_", zip)
      new_data[[col_name]] <- ifelse(input$zip_code == zip, 1, 0)
    }
    
    # Make prediction
    tryCatch({
      pred <- predict(rf_model, newdata = new_data)
      return(pred)
    }, error = function(e) {
      return(NA)
    })
  })
  
  # Render the prediction output with formatting
  output$prediction_output <- renderText({
    pred <- prediction()
    if (is.na(pred)) {
      return("Unable to make prediction")
    } else {
      formatted_price <- scales::dollar_format()(pred)
      return(paste(formatted_price))
    }
  })
  
  # Initialize the map with light theme
  output$map <- renderLeaflet({
    leaflet() %>%
      addProviderTiles(providers$CartoDB.Positron) %>%  # Light theme map tiles
      setView(lng = -118.2437, lat = 34.0522, zoom = 10) %>%  # Default to LA coordinates
      # Custom styling for the map
      htmlwidgets::onRender("
        function(el, x) {
          var map = this;
          map.getContainer().style.background = '#ffffff';
          map.getContainer().style.border = '1px solid #3d3d3d';
        }
      ")
  })
  
  # Update map when coordinates change with custom marker
  observe({
    coordinate_data <- coords()
    
    if (!is.null(coordinate_data)) {
      leafletProxy("map") %>%
        clearMarkers() %>%
        setView(lng = coordinate_data$lng, 
                lat = coordinate_data$lat, 
                zoom = 15) %>%
        addMarkers(lng = coordinate_data$lng, 
                   lat = coordinate_data$lat,
                   popup = input$address,
                   options = markerOptions(
                     riseOnHover = TRUE
                   ))
    }
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)