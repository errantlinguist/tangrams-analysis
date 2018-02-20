#!/usr/bin/env Rscript

# Plots mean token count for round and calculates.
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

#infile <- "D:\\Users\\tcshore\\Documents\\Projects\\Tangrams\\Data\\Analysis\\rounds_length.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

outfile <- args[2]

library(ggplot2)
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
# Hack to change legend label
names(df)[names(df) == "session"] <- "Dyad"
# https://stackoverflow.com/a/15665536
df$Dyad <- factor(df$Dyad, levels = paste(sort(as.integer(levels(df$Dyad)))))


#aggs <- aggregate(RA ~ Dyad * round, df, mean)

plot <- ggplot(df, aes(x=round, y=length))

#text(4, 7, expression(bar(x) == sum(frac(x[i], n), i==1, n)))
plot <- plot + xlab(expression(paste("Game round ", italic("i")))) + ylab("Token count")
aspectRatio <- 3/4
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio, plot.margin=margin(4,0,0,0), legend.background=element_rect(fill=alpha("white", 0.0)), legend.box.margin=margin(0,0,0,0), legend.box.spacing=unit(1, "mm"), legend.direction="horizontal", legend.margin=margin(0,0,0,0), legend.justification = c(0.99, 0.99), legend.position = c(0.99, 0.99), legend.text=element_text(family="mono", face="bold"))
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis") + scale_shape_manual(values=1:nlevels(df$Dyad))

plot <- plot + geom_point(aes(color=Dyad, shape=Dyad))
#plot <- plot + geom_line()
#plot
#plot <- plot + geom_jitter(alpha = 0.3, size=0.1)
#plot <- plot + geom_smooth(method="loess", level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
#regressionAlpha <- 0.333333
#print(sprintf("Using alpha transparency = %f for each individual regression line.", regressionAlpha), quote=FALSE)
plot <- plot + geom_smooth(method = "lm", formula = y ~ poly(x,2), level=0.95, fullrange=TRUE, size=0.7, color="darkred")
#plot <- plot + geom_smooth(method = "lm", formula = y ~ x, level=0.95, fullrange=TRUE, size=0.7, aes(color=Dyad))
plot <- plot + scale_y_log10()

xmin <- min(df$round)
xmax <- max(df$round)
#round_mra <- aggregate(MRA ~ round, data = df, FUN = mean)
ymin <- min(df$length)
ymax <- max(df$length)
#ymax <- 150
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