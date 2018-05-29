#!/usr/bin/env Rscript

# Plots RR for leaving out training data.
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

#infile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/removal-0-35.tsv"
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
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses = c(Condition="factor", Dyad="factor", round="integer", rank="integer", removal="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", removal="integer", session="factor", round="integer")))
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

df$TrainingSet <- 40 - df$removal
#df$TrainingSet <- factor(df$TrainingSet)

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

plot <- ggplot(df, aes(x=TrainingSet, y=RR, group=Condition, shape=Condition, color=Condition, linetype=Condition))
plot <- plot + xlab(expression(paste("Background data size (", italic("n"), " dialogs)"))) + ylab("Mean RR")
aspectRatio <- 9/16 # EMNLP 2018
#aspectRatio <- 21/36 # SemDial 2018
plot <- plot + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=aspectRatio, plot.margin=margin(4,0,0,0), legend.background=element_rect(fill=alpha("white", 0.0)), legend.box="vertical", legend.box.margin=margin(0,0,0,0), legend.box.spacing=unit(1, "mm"), legend.direction="horizontal", legend.margin=margin(0,0,0,0), legend.justification = c(0.99, 0.01), legend.position = c(0.99, 0.01), legend.text=element_text(family="mono", face="bold"), legend.title=element_blank()) 

# Manually created because viridis is annoying
#colors <- c("#21908CFF", "#440154FF")
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis", direction=-1) #+ scale_shape_manual(values=shapes)

#removal_mrrs_for_dyad_cond <- aggregate(RR ~ TrainingSet + Condition + Dyad, data = df, FUN = mean)
#removal_mrrs_for_dyad_cond <- within(removal_mrrs_for_dyad_cond,  ClusterCondition <- as.factor(paste(Condition, Dyad, TrainingSet, sep=",")))
#sapply(removal_mrrs_for_dyad_cond, class)
#kmeans_df <- removal_mrrs_for_dyad_cond[,c("RR","Dyad")]
#fit <- kmeans(kmeans_df, 4, nstart = 7)
# get cluster means
#aggregate(removal_mrrs_for_dyad_cond,by=list(fit$cluster),FUN=mean)
# append cluster assignment
#removal_mrrs_for_dyad_cond <- data.frame(removal_mrrs_for_dyad_cond, fit$cluster) 
#removal_mrrs_for_dyad_cond$fit.cluster <- as.factor(removal_mrrs_for_dyad_cond$fit.cluster)
#plot <- plot + geom_point(data = removal_mrrs_for_dyad_cond, aes(group=fit.cluster, color=Condition, shape=fit.cluster), size=0.3) + guides(shape=FALSE)


#agg_mrrs <- aggregate(RR ~ TrainingSet + Condition + Dyad, data = df, FUN = mean)
#plot <- plot + geom_line(aes(group=Condition, color=Condition))
#plot <- plot + geom_jitter(data=agg_mrrs, width = 0.5, height=0, size=0.3, aes(group=Dyad, color=Condition, shape=Condition)) + guides(shape=FALSE)

#plot <- plot + geom_area(stat = "summary", fun.y = "mean", position = position_dodge(width = 0.9), aes(group=Condition, color=Condition, shape=Condition, fill=Condition), reverse = TRUE)
#plot <- plot + geom_errorbar(stat = "summary", fun.data = "mean_sdl", fun.args = list(mult = 1), position =  position_dodge(width = 0.9), aes(group=Condition, color=Condition, shape=Condition))
plot <- plot + stat_summary(size=0.3, fun.data = mean_se, aes(group=Condition, color=Condition, shape=Condition))
agg_mrrs <- aggregate(RR ~ TrainingSet + Condition, data = df, FUN = mean)
plot <- plot + geom_line(data=agg_mrrs, aes(group=Condition, color=Condition))
#plot <- plot + geom_pointrange(stat = "summary", fun.data = "mean_se", aes(group=Condition, color=Condition), size=0.2)
#plot <- plot + stat_summary(size=0.3, fun.data = "mean_sdl", fun.args = list(mult = 1), aes(group=Condition, color=Condition, shape=Condition))
#plot <- plot + geom_errorbar(aes(group=Condition, color=Condition, shape=Condition, x=TrainingSet, y=RR, ymin=y-sd, ymax=y+sd))
#plot <- plot + geom_smooth(method = "lm", level=0.95, fullrange=TRUE, size=0.7, formula = y ~ poly(x, 2))
plot <- plot + scale_x_continuous(breaks=sort(unique(df$TrainingSet)), expand = c(0, 0))
ymin <- 0.4
ymax = 1.0
plot <- plot + coord_cartesian(ylim = c(ymin, ymax), expand = FALSE)

output_device <- file_ext(outfile)
print(sprintf("Writing plot to \"%s\" using format \"%s\".", outfile, output_device), quote=FALSE)

width <- 100
height <- width * aspectRatio
ggsave(outfile, plot = plot, device=output_device, width = width, height = height, units="mm", dpi=1000)
