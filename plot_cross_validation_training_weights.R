library(ggplot2)
library(lme4)
library(scales)

read_results <- function(inpath) {
  return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
}

# https://stackoverflow.com/a/27694724
windowsFonts(Times=windowsFont("Times New Roman"))

indir <- "D:\\Users\\tcshore\\Documents\\Projects\\Tangrams\\Data\\output\\tangrams-updating"
#setwd(indir)
#infiles <- list.files(pattern = "*bothspkr\\.tsv")
#infile_dfs = lapply(infiles, read.csv)
#do.call("rbind", list(DF1, DF2, DF3))

baseline_df <- read_results(file.path(indir, "Update0.0-bothspkr.tsv"))
updating_df <- read_results(file.path(indir, "Update1.0-bothspkr.tsv"))
df <- rbind(baseline_df, updating_df)
df$RR <- 1.0 / df$RANK
#df$UPDATE_WEIGHT <- ifelse(df$UPDATE_WEIGHT > 0, "yes", "no")
# Hack to change legend label
names(df)[names(df) == "UPDATE_WEIGHT"] <- "Weight"

model <- lmer(RR ~ ROUND + BACKGROUND_DATA_WORD_TOKEN_COUNT + INTERACTION_DATA_WORD_TOKEN_COUNT + (1|DYAD), data = df)

#refLevel <- 0
#Set the reference level for Training
#relevel(df$Weight, ref=refLevel) -> cvResults$Weight

df$fit <- predict(model)   #Add model fits to dataframe
model

# The palette with black:
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

plot <- ggplot(df, aes(x=ROUND, y=RR, group=Weight, shape=Weight, color=Weight, linetype=Weight)) + geom_jitter(alpha = 0.5)
plot <- plot + geom_smooth(method="lm", fullrange=TRUE)

xmin <- min(df$ROUND)
xmax <- round(max(df$ROUND), -1)
ymin <- 0
plot + xlab("Rank") + ylab("RR") + theme_bw() + theme(text=element_text(family="Times"), aspect.ratio=3/4) + scale_colour_manual(values=cbbPalette) + scale_x_continuous(limits=c(xmin, xmax), expand = c(0, 0), oob=squish, breaks = scales::pretty_breaks(n = 5)) +  scale_y_continuous(limits=c(ymin, 1.0), expand = c(0, 0), oob=squish)

# https://stackoverflow.com/a/31095291
#ggplot(tempEf,aes(TRTYEAR, r, group=interaction(site, Myc), col=site, shape=Myc )) + 
#  facet_grid(~N) +
#  geom_line(aes(y=fit, lty=Myc), size=0.8) +
#  geom_point(alpha = 0.3) + 
#  geom_hline(yintercept=0, linetype="dashed") +
#  theme_bw()