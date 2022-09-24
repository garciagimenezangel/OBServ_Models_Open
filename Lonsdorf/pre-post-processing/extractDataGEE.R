library(dplyr)
library(stringr)

######################################################
# FOLDERS AND OTHER SETTINGS
######################################################
# Clean session and source repo functions
rm(list = ls())

# Repositories
modelsRepo    = "C:/Users/Angel/git/OBServ/Observ_models//"
fieldDataRepo = "C:/Users/angel/git/OBServ/OBservData/"

# Folders
GEEdataDir   = paste0(modelsRepo, "data/GEE/GEE outputs/")
outputDir    = paste0(modelsRepo, "data/GEE/Processed/")
fieldDataDir = paste0(fieldDataRepo, "Final_Data/")

# Column names that will be in datasets and in the processed geodata tables
baseColNames = c("study_id", "site_id", "sampling_year") 
geeColNames  = c(baseColNames, "first") 

# Output file
outFile = paste0(Sys.Date(), ".csv")
outFile = paste0(outputDir, outFile)

# Other Settings
coordsDigits = 3 # Precision of the coordinates when doing JOIN operations between tables

######################################################
# GET DATA
######################################################
# Get field data
source(paste0(modelsRepo, "code/lib/functions.R")) # general functions for the repo
df = getOBServFieldData(fieldDataDir);

# Sampling year must be transformed into reference year
getRefYears = function(years) {
  if (length(years) == 1) {
    return(years[1])
  } else if (length(years) == 2) {
    return(years[2])   
  } 
  return(-1)
}
listYears        = str_split(df$sampling_year,"-")
refYears         = lapply(listYears, getRefYears)
df$sampling_year = as.character(refYears)

# Get GEE results data
geeFiles = list.files(GEEdataDir, full.names = FALSE, recursive = TRUE);
csvs     = str_detect(geeFiles, ".(csv)$"); # Take only csv's
geeFiles = geeFiles[csvs];

# Set GEE data ids
geeIDs = tools::file_path_sans_ext(geeFiles);
geeIDs = gsub("/", ".", geeIDs);         # replace slashes by points


######################################################
# COMPILE DATA INTO ONE SINGLE DATAFRAME
######################################################
# Output dataframe 
colNames      = c(baseColNames, geeIDs)
df_out        = data.frame(matrix(ncol = length(colNames), nrow = 0))
names(df_out) = colNames
df_temp       = df[baseColNames]
# Loop models
for (iGEE in seq(geeFiles)) {
  
  # Get model file and id
  geeID   = geeIDs[iGEE];
  geeFile = geeFiles[iGEE];
  
  # Read csv file
  geeData = read.csv(file = paste0(GEEdataDir, geeFile), header=T)
  
  # Some inconsistencies with refYear-sampling_year (my fault). Make sure that it has the column sampling_year
  if ("refYear" %in% colnames(geeData)) geeData$sampling_year = geeData$refYear
  geeData = geeData %>% dplyr::select(geeColNames)
  
  # Just in case there is duplicated data
  geeData = geeData[!duplicated(geeData),]
  
  # LEFT JOIN tables
  df_temp = merge(df_temp, geeData, all.x=TRUE, by=baseColNames)

  # Rename added column (the last one) in df_temp with geeID
  names(df_temp)[length(names(df_temp))] = geeID;
}

# Add geodata associated with this dataset, into the output dataframe
df_out = rbind(df_out, df_temp)

# Sort table by the two first columns (they should be "study_id" and "field_id")
ord = order(df_out[,1], as.numeric(df_out[,2]))
df_final = df_out[ord,]

######################################################
# SAVE RESULTS
######################################################
# Write csv 
write.csv(df_final, outFile, row.names=FALSE) 
print("Output table:")
print(outFile)

