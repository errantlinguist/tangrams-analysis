#!/usr/bin/env Rscript

# Plots RR for round and calculates significance using a general linear model with random effects for dyad ("sess")
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

infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

outfile <- args[2]

library(ggplot2)
library(lmerTest)
library(tools)  

if (!require("viridis")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("sjmgarnier/viridis")
  library(viridis)
}

read_results <- function(inpath) {
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", sess="factor", round="integer", rank="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))


df <- read_results(infile)
df$RR <- 1.0 / df$rank
#df$UPDATE_WEIGHT <- ifelse(df$UPDATE_WEIGHT > 0, "yes", "no")
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)
#reorder(levels(df$Condition), new.order=c("Baseline", "W", "U,W"))

refLevel <- "Baseline"
# Set the reference level
relevel(df$Condition, ref=refLevel) -> df$Condition

model <- lmer(RR ~ Condition + round + (1|sess), data = df, REML=TRUE)
summary(model)

plot <- ggplot(df, aes(x=round, y=RR, group=Condition, shape=Condition, color=Condition, linetype=Condition)) 
plot <- plot + stat_summary_bin(fun.data = mean_se, alpha=0.8, size=0.3)
#plot <- plot + geom_jitter(alpha = 0.3, size=0.1)
#plot <- plot + geom_smooth(method="loess", level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
plot <- plot + geom_smooth(method = "lm", formula = y ~ x + I(x^2), level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
plot <- plot + xlab("Round") + ylab("MRR") + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=1, plot.margin=margin(4,0,0,0), legend.background=element_rect(fill=alpha("white", 0.0)), legend.box.margin=margin(0,0,0,0), legend.box.spacing=unit(1, "mm"), legend.direction="horizontal", legend.margin=margin(0,0,0,0), legend.justification = c(0.99, 0.01), legend.position = c(0.99, 0.01), legend.text=element_text(family="mono", face="bold"), legend.title=element_blank()) 

# The palette with black:
#cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
#plot <- plot + scale_colour_manual(values=cbbPalette)
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis") #+ scale_colour_manual(values=cbbPalette)
#plot

xmin <- min(df$round)
#xmax <- round(max(df$round), digits = -1)
xmax <- max(df$round)
#round_mrrs <- aggregate(RR ~ round, data = df, FUN = mean)
#ymin <- min(round_mrrs)
ymin <- 0.4
ymax = 1.0
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)
#plot <- plot + scale_x_continuous(limits=c(xmin, xmax), expand = c(0, 0), breaks = scales::pretty_breaks(n = 5)) + scale_y_continuous(limits=c(ymin, 1.0), expand = c(0, 0))
#plot

output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)
ggsave(outfile, plot = plot, device=output_device, width = 100, height = 100, units="mm", dpi=1000)


# https://stackoverflow.com/a/31095291
#ggplot(tempEf,aes(TRTYEAR, r, group=interaction(site, Myc), col=site, shape=Myc )) + 
#  facet_grid(~N) +
#  geom_line(aes(y=fit, lty=Myc), size=0.8) +
#  geom_point(alpha = 0.3) + 
#  geom_hline(yintercept=0, linetype="dashed") +
#  theme_bw()