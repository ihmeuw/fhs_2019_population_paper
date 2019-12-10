library(ncdf4)
library(ncdf4.helpers)
require(data.table)

## Get the name of the value vars in the nc file
get_nc_value_name <- function(nc_file) {

  ## Get names
  nc_obj <- nc_open(nc_file)
  name<-names(nc_obj$var)

  ## Close file
  nc_close(nc_obj)

  ## Return the name
  return(name)

}

## Given value vars, convert nc file to R
xarray_nc_to_R <- function(nc_file, dimname = NULL, start=NA, count=NA, df_return = T) {

  if(is.null(dimname)){
    dimname <- get_nc_value_name(nc_file)
  }
  ## Open the file and show the attributes
  ncin <- nc_open(nc_file)
  print(ncin)

  ## Get the full array, using the variable name we want
  Rarray <- ncvar_get(ncin, dimname, start = start, count = count, collapse_degen=F)

  ## Get the fillvalue info
  fillvalue <- ncatt_get(ncin,dimname,"_FillValue")

  ## Get the dimension names in the right order
  array_dim <- ncdf4.helpers::nc.get.dim.names(ncin, dimname)

  ## Close the file
  nc_close(ncin)

  ## Get all of the dimension information in the order specified
  array_dim_list <- list()
  for(i in array_dim) {
    array_dim_list[[i]] <- ncin$dim[[i]]$vals
  }

  ## Fill in NaNs with NA
  Rarray[Rarray==fillvalue$value] <- NA


  ## Assign the dimension labels to the R array
  for(i in 1:length(array_dim_list)) {
    dimnames(Rarray)[[i]] <- array_dim_list[[i]]
  }

  ## Attach the dimension name to the array
  names(attributes(Rarray)$dimnames) <- array_dim

  if(df_return) {
    return(data.table(reshape2::melt(Rarray)))
  } else {
    return(Rarray)
  }
}
