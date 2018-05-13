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

# Set global numeric value formatting, even for e.g. model "summary(..)" values
options("scipen"=999, "digits"=5)
#infile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/results-cross-2-with-corefs.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

library(lmerTest)
library(optimx)

read_results <- function(inpath) {
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", session="factor", Referent="factor")))
}

df <- read_results(infile)
sapply(df, class)
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "rank"] <- "Rank"
names(df)[names(df) == "round"] <- "Round"
names(df)[names(df) == "session"] <- "Dyad"
names(df)[names(df) == "weight"] <- "MeanRA"
names(df)[names(df) == "words"] <- "Tokens"

df$RR <- 1.0 / df$Rank
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)

refLevel <- "Baseline"
# Set the reference level
relevel(df$Condition, ref=refLevel) -> df$Condition

# https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
control <- lmerControl(optimizer ="Nelder_Mead")
# Limited-memory BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS> in order to avoid failed convergence, caused by the addition of the "Tokens" condition
# See https://stats.stackexchange.com/a/243225/169001
#control <- lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B'))

print("Quadratic additive model with the condition \"RndAdt\":", quote=FALSE)
# NOTE: Eliminated because the RndAdt condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
m.additive <- lmer(RR ~ Adt + Wgt + RndAdt + scale(Tokens) + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.additive)

print("Quadratic additive model without the condition \"RndAdt\":", quote=FALSE)
# The RndAdt condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
# This is the final model from backwards selection: Removing any more effects significantly hurts model fit
m.additiveNoRndAdt <- lmer(RR ~ Adt + Wgt + scale(Tokens) + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.additiveNoRndAdt)

print("ANOVA comparison of quadratic additive model with and without \"RndAdt\" condition (to conclude that it is not significant):", quote=FALSE)
p <- anova(m.additive, m.additiveNoRndAdt)
format(p, digits=10)

print("Monomial additive model without the condition \"RndAdt\":", quote=FALSE)
m.monomialAdditiveNoRndAdt <- lmer(RR ~ Adt + Wgt + scale(Tokens) + Round + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.monomialAdditiveNoRndAdt)

print("ANOVA comparison of quadratic and monomial additive models, both without \"RndAdt\" condition (to conclude that the polynomial factor is significant):", quote=FALSE)
p <- anova(m.additiveNoRndAdt, m.monomialAdditiveNoRndAdt)
p

print("Quadratic additive model without the condition \"Wgt\" or \"RndAdt\":", quote=FALSE)
# The RndAdt condition does not improve fit, which means that the condition does not significantly affect reciprocal rank 
m.additiveNoRndAdtNoWgt <- lmer(RR ~ Adt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.additiveNoRndAdtNoWgt)

print("ANOVA comparison of quadratic additive model with and without \"Wgt\" condition (to conclude that it is significant):", quote=FALSE)
p <- anova(m.additiveNoRndAdt, m.additiveNoRndAdtNoWgt)
p

print("Quadratic additive model without the condition \"RndAdt\" or \"Tokens\":", quote=FALSE)
m.additiveNoRndAdtNoWords <- lmer(RR ~ Adt + Wgt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.additiveNoRndAdtNoWords)

print("ANOVA comparison of additive model without \"RndAdt\" and additive model without \"RndAdt\" or \"Tokens\" condition (to conclude that \"Tokens\" is significant):", quote=FALSE)
p <- anova(m.additiveNoRndAdt, m.additiveNoRndAdtNoWords)
p

print("Quadratic additive model without the condition \"RndAdt\" or \"Tokens\" but with \"Corefs\":", quote=FALSE)
m.additiveNoRndAdtCorefs <- lmer(RR ~ Adt + Wgt + poly(Round, 2) + Corefs + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.additiveNoRndAdtCorefs)
#0.0531 

print("Testing significance of relationship of \"Tokens\" with \"Round\" and \"Corefs\":", quote=FALSE)
# Only select the rows using just "Baseline"
df_baseline <- df[df$Condition %in% c("Baseline"), ]
m.tokensAdditive <- lmer(Tokens ~ poly(Round, 2) + Corefs + (1 + Referent | Dyad), data = df_baseline, REML = FALSE, control = control)
summary(m.tokensAdditive)

m.tokensInteractive <- lmer(Tokens ~ poly(Round, 2) * Corefs + (1 + Referent | Dyad), data = df_baseline, REML = FALSE, control = control)
summary(m.tokensInteractive)

p <- anova(m.tokensAdditive, m.tokensInteractive)
p

# INTERACTIVE MODELS ----------------------------------------------

print("Quadratic interaction model without the condition \"RndAdt\":", quote=FALSE)
m.interactionNoRndAdt <- lmer(RR ~ Adt * Wgt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
summary(m.interactionNoRndAdt)

print("ANOVA comparison of quadtratic additive model and interactive model, both without \"RndAdt\" condition (to conclude that there is no significant interaction):", quote=FALSE)
p <- anova(m.additiveNoRndAdt, m.interactionNoRndAdt)
p

#m.zeroModel <- lmer(RR ~ poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
#summary(m.zeroModel)

#m.noWgt <- lmer(RR ~ Adt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
#summary(m.noWgt)

#m.noAdaptation <- lmer(RR ~ Wgt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
#summary(m.noAdaptation)

# Does not fit data as well as one with Wgt as a fixed effect as well
#m.additiveNoWgt <- lmer(RR ~ Adt + RndAdt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
#summary(m.additiveComplex)

# Doesn't improve fit
#m.interactionComplex <- lmer(RR ~ Adt * Wgt + RndAdt + poly(Round, 2) + (1 + Adt * Wgt | Dyad), data = df, REML = FALSE, control = control)
#print("Most complex model:", quote=FALSE)
#summary(m.interactionComplex)

#m.interactionRandomSlope <- lmer(RR ~ Adt * Wgt + RndAdt + poly(Round, 2) + (1 + Adt + Wgt | Dyad), data = df, REML = FALSE, control = control)
#print("Most complex model:", quote=FALSE)
#summary(m.interactionComplex)


