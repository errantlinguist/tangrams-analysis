#!/usr/bin/env Rscript

# Calculates significance of different recoring methods for RR per round using a general linear model with random effects for dyad ("sess")
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
if(length(args) < 1)
{
  stop("Usage: <scriptname> INFILE")
}

infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

library(lmerTest)
library(texreg)

read_results <- function(inpath) {
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", Updating="logical", Weighting="logical", RandomData="logical")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
}

df <- read_results(infile)
df$RR <- 1.0 / df$rank
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "sess"] <- "Dyad"
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)
#reorder(levels(df$Condition), new.order=c("Baseline", "W", "U,W"))

refLevel <- "Baseline"
# Set the reference level
relevel(df$Condition, ref=refLevel) -> df$Condition

m.additive <- lmer(RR ~ Updating + Weighting + RandomData + poly(round, 2) + (1|Dyad), data = df, REML=FALSE)
m.interaction <- lmer(RR ~ Updating * Weighting + RandomData + poly(round, 2) + (1|Dyad), data = df, REML=FALSE)

#This is a test for whether the interaction model improved anything over the additive model.
p <- anova(m.additive, m.interaction)
p$Chisq
p$`Pr(>Chisq)`
p

summary(m.additive)
texreg(m.additive, digits=3, float.pos="!htb", single.row=TRUE)

summary(m.interaction)
texreg(m.interaction, digits=3, float.pos="!htb", single.row=TRUE)


print("Re-leveling to updating", quote=FALSE)
relevel(df$Condition, ref="Updating") -> df$Condition

m.additive <- lmer(RR ~ Updating + Weighting + RandomData + poly(round, 2) + (1|Dyad), data = df, REML=FALSE)
m.interaction <- lmer(RR ~ Updating * Weighting + RandomData + poly(round, 2) + (1|Dyad), data = df, REML=FALSE)

#This is a test for whether the interaction model improved anything over the additive model.
p <- anova(m.additive, m.interaction)
p$Chisq
p$`Pr(>Chisq)`
p

summary(m.additive)
texreg(m.additive, digits=3, float.pos="!htb", single.row=TRUE)

summary(m.interaction)
texreg(m.interaction, digits=3, float.pos="!htb", single.row=TRUE)