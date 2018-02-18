#!/usr/bin/env Rscript

library(ggplot2)
library(lmerTest)

if (!require("viridis")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("sjmgarnier/viridis")
  library(viridis)
}

read_results <- function(inpath) {
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
}

# https://stackoverflow.com/a/27694724
try(windowsFonts(Times=windowsFont("Times New Roman")))

#indir <- "D:\\Users\\tcshore\\Documents\\Projects\\Tangrams\\Data\\output\\tangrams-updating"
indir <- "/home/tshore/Projects/tangrams-restricted/Data/Analysis"
#setwd(indir)
#infiles <- list.files(pattern = "*bothspkr\\.tsv")
#infile_dfs = lapply(infiles, read.csv)
#do.call("rbind", list(DF1, DF2, DF3))

#df <- read_results(file.path(indir, "results.csv"))
df <- read_results(file.path(indir, "weighting.csv"))
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
plot <- plot + geom_smooth(method="glm", level=0.95, fullrange=TRUE, size=0.7, alpha=0.2)
plot <- plot + xlab("Round") + ylab("MRR") + theme_light() + theme(text=element_text(family="Times"), aspect.ratio=1, plot.margin=margin(0,0,0,0), legend.margin=margin(0,0,0,0), legend.box.margin=margin(0,0,0,0)) 

# The palette with black:
#cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
#plot <- plot + scale_colour_manual(values=cbbPalette)
plot <- plot + scale_color_viridis(discrete=TRUE, option="viridis") #+ scale_colour_manual(values=cbbPalette)
plot

xmin <- min(df$round)
#xmax <- round(max(df$round), digits = -1)
xmax <- max(df$round)
#round_mrrs <- aggregate(RR ~ round, data = df, FUN = mean)
#ymin <- min(round_mrrs)
ymin <- 0.4
ymax = 1.0
plot <- plot + coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax), expand = FALSE)
#plot <- plot + scale_x_continuous(limits=c(xmin, xmax), expand = c(0, 0), breaks = scales::pretty_breaks(n = 5)) + scale_y_continuous(limits=c(ymin, 1.0), expand = c(0, 0))
plot

#outpath <- "D:\\Users\\tcshore\\Downloads\\fig-weighting.pdf"
outpath <- file.path(indir, "round-mrr-weighting.pdf")
ggsave(outpath, plot = plot, device="pdf", width = 100, height = 100, units="mm", dpi=1000)


# https://stackoverflow.com/a/31095291
#ggplot(tempEf,aes(TRTYEAR, r, group=interaction(site, Myc), col=site, shape=Myc )) + 
#  facet_grid(~N) +
#  geom_line(aes(y=fit, lty=Myc), size=0.8) +
#  geom_point(alpha = 0.3) + 
#  geom_hline(yintercept=0, linetype="dashed") +
#  theme_bw()