rm(list=ls())
library(rlist)
library(stringr)

data_folder = "C:/Users/angel/git/OBservData/Final_Data/"
# data_folder = "C:/Users/angel.gimenez/git/OBservData/Final_Data/"
dfOBServFieldData = read.csv(file=paste0(data_folder,"CropPol_field_level_data.csv"), header = TRUE)
dfOBServFieldData  = dfOBServFieldData[, c("study_id", "site_id", "latitude","longitude", "management", "sampling_start_month", "sampling_end_month", "sampling_year")]

# # Some site_id's = NA. Replace by "" (I think site_id=NA is never the case anymore)
# dfOBServFieldData$site_id <- unlist(lapply(dfOBServFieldData$site_id, as.character))
# dfOBServFieldData$site_id[is.na(dfOBServFieldData$site_id)] = "noID"

# remove NA locations
dfOBServFieldData = dfOBServFieldData[!is.na(dfOBServFieldData$latitude) & !is.na(dfOBServFieldData$longitude),]

# Derive start and end date from data columns
listYears     = str_split(dfOBServFieldData$sampling_year,"-")
arrStartMonth = dfOBServFieldData$sampling_start_month
arrEndMonth   = dfOBServFieldData$sampling_end_month
startDate     = c()
endDate       = c()
refYear       = c()
for (i in seq(1,length(listYears))) {
  years      = listYears[[i]]
  if (length(years) == 1) {
    startYear = years[1]
    endYear   = years[1]
  } else if (length(years) == 2) {
    startYear = years[1]
    endYear   = years[2]    
  } else {
    print("ERROR")
    break
  }
  startMonth = ifelse(is.na(arrStartMonth[i]), "1",arrStartMonth[i]) 
  endMonth   = ifelse(is.na(arrEndMonth[i]), "12",arrEndMonth[i])
  startDate  = c(startDate, paste0(startYear, "-", str_pad(startMonth, 2, pad="0"), "-01"))
  endDate    = c(endDate, paste0(endYear, "-", str_pad(endMonth, 2, pad="0"), "-01"))
  refYear    = c(refYear, endYear)
}

# Select management="organic"
organic = !is.na(dfOBServFieldData$management) & (dfOBServFieldData$management == "organic")

# Save sites
sites           = dfOBServFieldData[organic, c("study_id","site_id","latitude","longitude")]
sites$startDate = startDate[organic]
sites$endDate   = endDate[organic]
sites$refYear   = refYear[organic]
sites_file  = "C:/Users/angel/git/OBServ_models/data/sitesOrganic.csv"
# sites_file  = "C:/Users/angel.gimenez/git/OBServ_models/data/sitesOrganic.csv"
write.csv(sites, sites_file, row.names=FALSE)
