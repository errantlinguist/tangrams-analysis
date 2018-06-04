#!/usr/bin/env Rscript

# Plots mean token count for round.
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
  #return(read.csv(inpath, sep = "\t", colClasses = c(session="factor", round="integer", rank="integer", length="integer")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", session="factor", round="integer")))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))

df <- read_results(infile)
sapply(df, class)
# Only select the rows using just "Baseline"
df <- df[df$cond %in% c("Baseline"), ]
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "rank"] <- "Rank"
names(df)[names(df) == "round"] <- "Round"
names(df)[names(df) == "session"] <- "Dyad"
names(df)[names(df) == "weight"] <- "RA"
names(df)[names(df) == "words"] <- "Tokens"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))


plot <- ggplot(df, aes(x=Round, y=Tokens))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Token count")
aspectRatio <- 9/16
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio, plot.margin=margin(12,0,0,0))

#break_datapoints <- df[df$Round %% 5 == 0, ]
#plot <- plot + stat_summary(data = break_datapoints, fun.data = mean_se, size=0.3)
plot <- plot + stat_summary(fun.data = mean_se, size=0.2)
#plot <- plot + geom_point(size=0.3, aes(group=Dyad, color=Dyad)) + guides(color=FALSE)
plot <- plot + geom_smooth(method = "lm", formula = y ~ poly(x,2), level=0.95, fullrange=TRUE, size=0.7, color="darkred")
#plot <- plot + scale_y_log10()


xmin <- 1
#xmax <- max(df$Round)
xmax <- 80
print(sprintf("Plotting round %d to %d.", xmin, xmax), quote=FALSE)
xb <- seq(xmin, xmax)
xb <- subset(xb, (xb %% 20 == 0) | (xb == xmin) | (xb == xmax))
print(xb)
plot <- plot + scale_x_continuous(breaks = xb, expand = c(0, 0))

ymin <- min(df$Tokens)
#ymax <- max(df$Tokens)
ymax <- 50
print(sprintf("Plotting token count %d to %d.", ymin, ymax), quote=FALSE)
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)


output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
width <- 100 # EMNLP 2018
#width <- 80 # SemDial 2018
height <- width * aspectRatio
ggsave(outfile, plot = plot, device=output_device, width = width, height = height, units="mm", dpi=1000)
