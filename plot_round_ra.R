#!/usr/bin/env Rscript

# Plots mean referring ability (RA) weight for round and calculates significance using a general linear model with random effects for dyad ("session").
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

#infile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/results-cross-2.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

outfile <- args[2]

library(ggplot2)
library(tools)

read_results <- function(inpath) {
  #return(read.csv(inpath, sep = "\t", colClasses = c(Dyad="factor", round="integer", role="factor", word="factor")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", session="factor", round="integer")))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))

df <- read_results(infile)
sapply(df, class)
# Only select the rows using just RA weighting "Wgt"
df <- df[df$cond %in% c("Wgt"), ]
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "rank"] <- "Rank"
names(df)[names(df) == "round"] <- "Round"
names(df)[names(df) == "session"] <- "Dyad"
names(df)[names(df) == "weight"] <- "RA"
names(df)[names(df) == "words"] <- "Tokens"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))


plot <- ggplot(df, aes(x=Round, y=RA))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Mean RA")
aspectRatio <- 9/16
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio, plot.margin=margin(12,0,0,0))

#break_datapoints <- df[df$Round %% 5 == 0, ]
#plot <- plot + stat_summary(data = break_datapoints, fun.data = mean_se, size=0.3)
plot <- plot + stat_summary(fun.data = mean_se, size=0.2)
plot <- plot + geom_smooth(method = "lm", formula = y ~ poly(x,2), level=0.95, fullrange=TRUE, size=0.7, color="darkred")


xmin <- 1
#xmax <- max(df$Round)
xmax <- 80
print(sprintf("Plotting round %d to %d.", xmin, xmax), quote=FALSE)
xb <- seq(xmin, xmax)
xb <- subset(xb, (xb %% 20 == 0) | (xb == xmin) | (xb == xmax))
print(xb)
plot <- plot + scale_x_continuous(breaks = xb, expand = c(0, 0))

ymin <- 0.05
#ymax <- max(df$RA)
ymax <- 0.3
print(sprintf("Plotting RA %f to %f.", ymin, ymax), quote=FALSE)
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)


output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
width <- 100 # EMNLP 2018
#width <- 80 # SemDial 2018
height <- width * aspectRatio
ggsave(outfile, plot = plot, device=output_device, width = width, height = height, units="mm", dpi=1000)
