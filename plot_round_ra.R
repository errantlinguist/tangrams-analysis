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


#aggs <- aggregate(RA ~ Dyad * round, df, mean)

plot <- ggplot(df, aes(x=Round, y=RA))
#plot

#text(4, 7, expression(bar(x) == sum(frac(x[i], n), i==1, n)))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Mean RA")
aspectRatio <- 3/4
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio)

plot <- plot + stat_summary(fun.data = mean_se, size=0.01)
#plot <- plot + geom_line()
#plot
#plot <- plot + geom_jitter(alpha = 0.3, size=0.1)
#plot <- plot + geom_smooth(method="loess", level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
#regressionAlpha <- 0.333333
#print(sprintf("Using alpha transparency = %f for each individual regression line.", regressionAlpha), quote=FALSE)
plot <- plot + geom_smooth(method = "lm", formula = y ~ poly(x,2), level=0.95, fullrange=TRUE, size=0.7, color="darkred")
#plot <- plot + geom_smooth(method = "lm", formula = y ~ x, level=0.95, fullrange=TRUE, size=0.7, aes(color=Dyad))
#plot

xmin <- min(df$Round)
xmax <- max(df$Round)
#round_mra <- aggregate(MRA ~ Round, data = df, FUN = mean)
ymin <- 0
#ymax <- max(round_mra$MRA)
ymax <- 0.5
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)
#plot <- plot + scale_x_continuous(limits=c(xmin, xmax), expand = c(0, 0), breaks = scales::pretty_breaks(n = 5)) + scale_y_continuous(limits=c(ymin, 1.0), expand = c(0, 0))
#plot

output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
width <- 100
height <- width * aspectRatio
ggsave(outfile, plot = plot, device=output_device, width = width, height = height, units="mm", dpi=1000)

# NOTE: This library causes problems with plotting aesthetics using "alpha(..)" function because it redefines it <https://github.com/const-ae/ggsignif/issues/2>
#library(psych)