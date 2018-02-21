#!/usr/bin/env Rscript

# Calculates significance of different rescoring methods for RR per round using a general linear model with random effects for dyad ("sess").
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

#infile <- "/home/tshore/Projects/tangrams-restricted/Data/Analysis//update-weight-3.tsv"
#infile <- "D:\\Users\\tcshore\\Documents\\Projects\\Tangrams\\Data\\Analysis\\update-weight-3.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

library(lmerTest)
#library(texreg)

read_results <- function(inpath) {
  #return(read.xlsx2(inpath, 1, colClasses = c(cond="factor", sess="factor", round="integer", rank="integer", mrr="numeric", accuracy="integer")))
  #return(read.csv(inpath, sep = "\t"))
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", sess="factor")))
  #return(read.csv(inpath, sep = "\t", colClasses=c(DYAD="factor", ONLY_INSTRUCTOR="logical", WEIGHT_BY_FREQ="logical", UPDATE_WEIGHT="factor")))
}

df <- read_results(infile)
sapply(df, class)
df$RR <- 1.0 / df$rank
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "sess"] <- "Dyad"
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)
#reorder(levels(df$Condition), new.order=c("Baseline", "W", "U,W"))

refLevel <- "Baseline"
# Set the reference level
relevel(df$Condition, ref=refLevel) -> df$Condition

print("Additive model with the condition \"Random\":", quote=FALSE)
# NOTE: Eliminated because the Random condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
m.additive <- lmer(RR ~ Updating + Weighting + Random + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
summary(m.additive)

print("Additive model without the condition \"Random\":", quote=FALSE)
# The Random condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
# This is the final model from backwards selection: Removing any more effects significantly hurts model fit
m.additiveNoRandomCondition <- lmer(RR ~ Updating + Weighting + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
summary(m.additiveNoRandomCondition)
#texreg(m.additiveNoRandomCondition, single.row=TRUE, float.pos="htb", digits=3, fontsize="small")

print("ANOVA comparison of additive model with and without \"Random\" condition (to conclude that it is not significant):", quote=FALSE)
p <- anova(m.additive, m.additiveNoRandomCondition)
p
#summary(p)

print("Interaction model without the condition \"Random\":", quote=FALSE)
# The Random condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
# This is the final model from backwards selection: Removing any more effects significantly hurts model fit
m.interactionNoRandomCondition <- lmer(RR ~ Updating * Weighting + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
summary(m.interactionNoRandomCondition)

print("ANOVA comparison of additive model and interactive model, both without \"Random\" condition (to conclude that there is no significant interaction):", quote=FALSE)
p <- anova(m.additiveNoRandomCondition, m.interactionNoRandomCondition)
p
#summary(p)

#m.zeroModel <- lmer(RR ~ poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
#summary(m.zeroModel)

#m.noWeighting <- lmer(RR ~ Updating + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
#summary(m.noWeighting)

#m.noUpdating <- lmer(RR ~ Weighting + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
#summary(m.noUpdating)

# Does not fit data as well as one with Weighting as a fixed effect as well
#m.additiveNoWeighting <- lmer(RR ~ Updating + Random + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
#summary(m.additiveComplex)

# Doesn't improve fit
#m.interactionComplex <- lmer(RR ~ Updating * Weighting + Random + poly(round, 2) + (1 + Updating * Weighting | Dyad), data = df, REML=FALSE)
#print("Most complex model:", quote=FALSE)
#summary(m.interactionComplex)

#m.interactionRandomSlope <- lmer(RR ~ Updating * Weighting + Random + poly(round, 2) + (1 + Updating + Weighting | Dyad), data = df, REML=FALSE)
#print("Most complex model:", quote=FALSE)
#summary(m.interactionComplex)


