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

#args <- commandArgs(trailingOnly=TRUE)
#if(length(args) < 2)
#{
#  stop("Usage: <scriptname> INFILE OUTFILE")
#}

infile <- "/home/tshore/Projects/tangrams-restricted/Data/Analysis/rounds_length.csv"
#infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

#outfile <- args[2]

library(ggplot2)
#library(texreg)
library(tools)  

if (!require("viridis")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("sjmgarnier/viridis")
  library(viridis)
}

read_results <- function(inpath) {
  return(read.csv(inpath, sep = "\t", colClasses = c(session="factor", round="integer", rank="integer", length="integer")))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))

df <- read_results(infile)
df$RR <- 1.0 / df$rank
df$MRA <- df$weight / df$length
# Hack to change legend label
names(df)[names(df) == "session"] <- "Dyad"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))

plot <- ggplot(df, aes(x=round, y=MRA))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Mean RA")
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=1, plot.margin=margin(4,0,0,0), legend.background=element_rect(fill=alpha("white", 0.0)), legend.box.margin=margin(0,0,0,0), legend.box.spacing=unit(1, "mm"), legend.direction="horizontal", legend.margin=margin(0,0,0,0), legend.justification = c(0.99, 0.99), legend.position = c(0.99, 0.99), legend.text=element_text(family="mono", face="bold"))
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis") + scale_shape_manual(values=1:nlevels(df$Dyad))

plot <- plot + stat_summary_bin(fun.data = mean_se, alpha=0.8, size=0.3, aes(group=Dyad, color=Dyad, shape=Dyad))
#plot <- plot + geom_line()
plot
#plot <- plot + geom_jitter(alpha = 0.3, size=0.1)
#plot <- plot + geom_smooth(method="loess", level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
plot <- plot + geom_smooth(method = "lm", formula = y ~ x, level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
plot

xmin <- min(df$round)
xmax <- max(df$round)
round_mra <- aggregate(MRA ~ round, data = df, FUN = mean)
ymin <- 0
#ymax <- max(round_mra$MRA)
ymax <- 0.3
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)
#plot <- plot + scale_x_continuous(limits=c(xmin, xmax), expand = c(0, 0), breaks = scales::pretty_breaks(n = 5)) + scale_y_continuous(limits=c(ymin, 1.0), expand = c(0, 0))
plot

output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
ggsave(outfile, plot = plot, device=output_device, width = 100, height = 100, units="mm", dpi=1000)

# NOTE: This library causes problems with plotting aesthetics using "alpha(..)" function because it redefines it <https://github.com/const-ae/ggsignif/issues/2>
library(psych)