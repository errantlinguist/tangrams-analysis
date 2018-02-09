library(ggplot2)

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

model <- lm(ROUND ~ RR, data = df)
summary(model)

# + scale_color_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73","#F0E442", "#0072B2", "#D55E00", "#CC79A7"))
scatter_plot <- ggplot(df, aes(x=ROUND, y=RR, group=UPDATE_WEIGHT, shape=UPDATE_WEIGHT, color=UPDATE_WEIGHT, linetype=UPDATE_WEIGHT)) + geom_jitter(alpha = 0.5) + xlab("Rank") + ylab("RR") + theme_light() + theme(text=element_text(family="Times"))
scatter_plot + geom_smooth(method="lm")



