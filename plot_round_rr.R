#!/usr/bin/env Rscript

# Plots RR for round and calculates significance using a general linear model with random effects for dyad ("sess").
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

outfile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/round-rr.pdf"
outfile <- args[2]

library(ggplot2)
library(tools)  

if (!require("viridis")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("sjmgarnier/viridis")
  library(viridis)
}

read_results <- function(inpath) {
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", sess="factor", round="integer", rank="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", session="factor", round="integer")))
}

std.error <- function(x) {
  # https://www.rdocumentation.org/packages/plotrix/versions/3.7/topics/std.error
  return(sd(x)/sqrt(sum(!is.na(x))))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))

print(sprintf("Reading data from \"%s\".", infile), quote=FALSE)
df <- read_results(infile)
sapply(df, class)
# Only select the rows using just baseline and RA weighting "Wgt" (SemDial paper)
#df <- df[df$cond %in% c("Baseline", "Wgt"), ]
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "rank"] <- "Rank"
names(df)[names(df) == "round"] <- "Round"
names(df)[names(df) == "session"] <- "Dyad"
names(df)[names(df) == "weight"] <- "RA"
names(df)[names(df) == "words"] <- "Tokens"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))

df$RR <- 1.0 / df$Rank
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)

refLevel <- "Baseline"
# Set the reference level
relevel(df$Condition, ref=refLevel) -> df$Condition

rank_digits <- 5
print(sprintf("Printing ranks rounded to %d significant figures.", rank_digits), quote=FALSE)

print("Condition avg rank:", quote=FALSE)
print(aggregate(Rank ~ Condition, data = df, FUN = mean), short=FALSE, digits=rank_digits)
print("Condition avg rank standard deviation:", quote=FALSE)
print(aggregate(Rank ~ Condition, data = df, FUN = sd), short=FALSE, digits=rank_digits)
print("Condition avg rank standard error:", quote=FALSE)
print(aggregate(Rank ~ Condition, data = df, FUN = std.error), short=FALSE, digits=rank_digits)

mrr_digits <- 4
print(sprintf("Printing MRR rounded to %d significant figures.", mrr_digits), quote=FALSE)

print("Condition MRR:", quote=FALSE)
print(aggregate(RR ~ Condition, data = df, FUN = mean), short=FALSE, digits=mrr_digits)
print("Condition MRR standard deviation:", quote=FALSE)
print(aggregate(RR ~ Condition, data = df, FUN = sd), short=FALSE, digits=mrr_digits)
print("Condition MRR standard error:", quote=FALSE)
print(aggregate(RR ~ Condition, data = df, FUN = std.error), short=FALSE, digits=mrr_digits)

plot <- ggplot(df, aes(x=Round, y=RR, group=Condition, shape=Condition, color=Condition, linetype=Condition))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Mean RR")
#aspectRatio <- 3/4 # EMNLP 2018
#aspectRatio <- 2/6 # Semdial 2018
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio, plot.margin=margin(12,0,0,0), legend.background=element_rect(fill=alpha("white", 0.0)), legend.box.margin=margin(0,0,0,0), legend.box.spacing=unit(1, "mm"), legend.direction="horizontal", legend.margin=margin(0,0,0,0), legend.justification = c(0.99, 0.01), legend.position = c(0.99, 0.01), legend.text=element_text(family="mono", face="bold"), legend.title=element_blank()) 
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis", direction=-1)


break_datapoints <- df[df$Round %% 5 == 0, ]
plot <- plot + stat_summary(data = break_datapoints, fun.data = mean_se, size=0.3)
plot <- plot + geom_smooth(method = "lm", formula = y ~ poly(x, 2), level=0.95, fullrange=TRUE, size=0.7)

xmin <- 1
#xmax <- max(df$Round)
xmax <- 80
print(sprintf("Plotting round %d to %d.", xmin, xmax), quote=FALSE)
xb <- seq(xmin, xmax)
xb <- subset(xb, (xb %% 20 == 0) | (xb == xmin) | (xb == xmax))
plot <- plot + scale_x_continuous(breaks = xb, expand = c(0, 0))

ymin <- 0.40
#ymax <- max(df$RA)
ymax <- 1
print(sprintf("Plotting RR %f to %f.", ymin, ymax), quote=FALSE)
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)


output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
width <- 100 # EMNLP 2018
#width <- 160 # SemDial 2018
height <- width * aspectRatio
ggsave(outfile, plot = plot, device=output_device, width = width, height = height, units="mm", dpi=1000)
