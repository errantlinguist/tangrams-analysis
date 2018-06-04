#!/usr/bin/env Rscript

# Calculates ROC AUC.
#
#
# Copyright 2018 Todd Shore
#
#	Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

args <- commandArgs(trailingOnly=TRUE)
if(length(args) < 2)
{
  stop("Usage: <scriptname> INFILE OUTFILE")
}

#infile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/roc-curve.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

outfile <- args[2]


read_results <- function(inpath) {
  #return(read.csv(inpath, sep = "\t", colClasses = c(session="factor", round="integer", rank="integer", length="integer")))
  return(read.csv(inpath, sep = "\t", colClasses = c(EVAL_SESSION="factor", IS_INSTRUCTOR="logical", IS_TARGET="logical", ROUND="integer")))
}


df <- read_results(infile)
sapply(df, class)
names(df)[names(df) == "EVAL_SESSION"] <- "Dyad"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))


require(pROC)
df_roc <- by(df, df$WORD, function(x) roc(x$IS_TARGET, x$PROB_TRUE, plot = FALSE))
sapply(df_roc, class)
df_roc["the"]
#aggregate(roc(df$IS_TARGET, df$PROB_TRUE, plot = FALSE) ~ df$WORD, FUN = mean)

#df_roc <- aggregate(x = df["date"], 
#                      by = list(month = substr(dates$date, 1, 7)), 
#                      FUN = max)

#do.call(rbind,
#        by(df, df$WORD, function(x) cbind(x[1,c("code","index")],value=min(x$value)))
#)