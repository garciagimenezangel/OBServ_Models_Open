library(dplyr)
library(stringr)
rm(list=ls())

#######################################
# Load the data (models and cropPol)
#######################################
modelsRepo = "C:/Users/Angel/git/OBServ/Observ_models//"
df_model   = read.csv(paste0(modelsRepo, "data/model_data.csv"), header=T) %>% select(!contains("geodata"))

fieldDataRepo = "C:/Users/angel/git/OBServ/OBservData/"
fieldDataDir = paste0(fieldDataRepo, "Final_Data/")
source(paste0(modelsRepo, "code/lib/functions.R")) # general functions for the repo
df_field = getOBServFieldData(fieldDataDir) %>% select(c('study_id', 'site_id','management'))

df_merged = merge(df_model, df_field, by=c('study_id', 'site_id'))

##################
# Process columns 
##################
processModelCols = function(df) {
  # 0. Replace management=NA by "conventional"
  df[is.na(df$management), "management" ] = "conventional"
  # 1. At sites where organic farming is practised, replace model values with the model values obtained with "management" bonus 
  col_small   = colnames(df)[str_detect(colnames(df), "Lonsdorf.small")]
  model_names = sapply(str_split(col_small, "Lonsdorf.small_"), "[[", 2) 
  for (model in model_names) {
    # small
    modelConventional                 = paste0("Lonsdorf.small_",model)
    modelOrganic                      = paste0("Lonsdorf.organic.small_",model)
    selection                         = (df$management == "organic") & !is.na(df[,modelOrganic]) # select values on organic management and not NA in the model
    df[selection , modelConventional] = df[selection , modelOrganic]                             # replace values and drop column organic management
    # large
    modelConventional                 = paste0("Lonsdorf.large_",model)
    modelOrganic                      = paste0("Lonsdorf.organic.large_",model)
    selection                         = (df$management == "organic") & !is.na(df[,modelOrganic]) # select values on organic management and not NA in the model
    df[selection , modelConventional] = df[selection , modelOrganic]                             # replace values and drop column organic management
  }
  # sanity check. All must be zero's or NAs. 
  a = df[ (df$management == "organic"), paste0("Lonsdorf.small_",model_names)] - df[df$management == "organic" ,paste0("Lonsdorf.organic.small_",model_names)]
  b = df[df$management == "organic" ,paste0("Lonsdorf.large_",model_names)] - df[df$management == "organic" ,paste0("Lonsdorf.organic.large_",model_names)]
  print("Sanity check:")
  print(all(a == 0, na.rm = T))
  print(all(b == 0, na.rm = T))
  df = df %>% select(!contains("organic")) # drop models with organic (values already used)
  # 2. Combined models guilds large and small
  col_small   = colnames(df)[str_detect(colnames(df), "Lonsdorf.small")]
  col_large   = colnames(df)[str_detect(colnames(df), "Lonsdorf.large")]
  model_names = sapply(str_split(col_small, "Lonsdorf.small_"), "[[", 2) 
  for (model in model_names) {
    new_model_name = paste0("Lonsdorf.",model)
    df[ ,new_model_name] = df[ , paste0("Lonsdorf.small_",model)]*0.5 + 
      df[ , paste0("Lonsdorf.large_",model)]*0.5
  }
  df = df %>% select(!contains("small")) 
  df = df %>% select(!contains("large"))
  return(df)
}

df_merged = processModelCols(df_merged)
write.csv(df_merged, file=paste0(modelsRepo,"data/model_data_lite.csv"), row.names = F)



