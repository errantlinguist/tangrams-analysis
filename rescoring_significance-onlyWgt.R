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
#infile <- "~/Projects/tangrams-restricted/Data/Analysis/2018-04-27/results-cross-2-with-corefs-metadata.tsv"
infile <- args[1]
if (!file_test("-f", infile)) {
  stop(sprintf("No file found at \"%s\".", infile));
}

library(lmerTest)
library(optimx)
#library(stargazer)

read_results <- function(inpath) {
  return(read.csv(inpath, sep = "\t", colClasses = c(cond="factor", session="factor", Referent="factor")))
}

df <- read_results(infile)
sapply(df, class)
# Only select the rows using just baseline and RA weighting "Wgt" (SemDial paper)
df <- df[df$cond %in% c("Baseline", "Wgt"), ]
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
#control <- lmerControl(optimizer ="Nelder_Mead")
# Limited-memory BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS> in order to avoid failed convergence, caused by the addition of the "Tokens" condition
# See https://stats.stackexchange.com/a/243225/169001
#control <- lmerControl(optimizer ='optimx', optCtrl=list(method='nlm'))

print("Quadratic additive model:", quote=FALSE)
# This is the final model from backwards selection: Removing any more effects significantly hurts model fit
m.additive <- lmer(RR ~ + Wgt + scale(Tokens) + poly(Round, 2) + (1 + Wgt | Dyad ), data = df, REML = FALSE)
summary(m.additive)

print("Quadratic additive model with manipulator gender:", quote=FALSE)
m.additiveGender <- lmer(RR ~ + Wgt + scale(Tokens) + poly(Round, 2) + ManipulatorGENDER + (1 + Wgt + ManipulatorGENDER | Dyad ), data = df, REML = FALSE)
summary(m.additiveGender)

print("ANOVA comparison of additive model with and without manipulator gender (to conclude that gender is not significant):", quote=FALSE)
anova(m.additive, m.additiveGender)

print("Monomial additive model:", quote=FALSE)
m.monomialAdditive <- lmer(RR ~ Wgt + scale(Tokens) + Round + (1 + Wgt | Dyad), data = df, REML = FALSE)
summary(m.monomialAdditive)

print("ANOVA comparison of quadratic and monomial additive models (to conclude that the polynomial factor is significant):", quote=FALSE)
p <- anova(m.additive, m.monomialAdditive)
p

print("Quadratic additive model without the condition \"Tokens\":", quote=FALSE)
m.additiveNoTokens <- lmer(RR ~ Wgt + poly(Round, 2) + (1 + Wgt | Dyad), data = df, REML = FALSE)
summary(m.additiveNoTokens)

print("ANOVA comparison of additive model and additive model without \"Tokens\" condition (to conclude that \"Tokens\" is significant):", quote=FALSE)
p <- anova(m.additive, m.additiveNoTokens)
p

print("Quadratic additive model with the condition \"Corefs\":", quote=FALSE)
m.additiveCorefs <- lmer(RR ~ + Wgt + poly(Round, 2) + scale(Tokens) + Corefs + (1 + Wgt | Dyad ), data = df, REML = FALSE)
summary(m.additiveCorefs)
#0.0531 

print("ANOVA comparison of additive model and additive model with and without \"Corefs\" (to conclude that \"Corefs\" is not significant):", quote=FALSE)
p <- anova(m.additive, m.additiveCorefs)
format(p)

print("Testing significance of relationship of \"Tokens\" with \"Round\" and \"Corefs\":", quote=FALSE)
# Only select the rows using just "Baseline"
df_baseline <- df[df$Condition %in% c("Baseline"), ]
m.tokensAdditive <- lmer(Tokens ~ poly(Round, 2) + Corefs + (1  | Dyad), data = df_baseline, REML = FALSE)
summary(m.tokensAdditive)

m.tokensInteractive <- lmer(Tokens ~ poly(Round, 2) * Corefs + (1 | Dyad), data = df_baseline, REML = FALSE)
summary(m.tokensInteractive)

p <- anova(m.tokensAdditive, m.tokensInteractive)
p

# INTERACTIVE MODELS ----------------------------------------------

print("Quadratic interaction model with the condition \"Corefs\":", quote=FALSE)
m.interactionCorefs.full <- lmer(RR ~ Wgt * poly(Round, 2) * scale(Tokens) * Corefs + (1 + Wgt | Dyad), data = df, REML = FALSE)
summary(m.interactionCorefs.full)

m.interactionCorefs.partial <- lmer(RR ~ Wgt + poly(Round, 2) + scale(Tokens) + Corefs + Wgt:poly(Round, 2) + Wgt:scale(Tokens) + poly(Round, 2):scale(Tokens) + poly(Round, 2):scale(Tokens):Corefs + (1 + Wgt | Dyad), data = df, REML = FALSE)
summary(m.interactionCorefs.partial)

print("ANOVA comparison of quadtratic additive model and partial interactive model (to conclude there is significant interaction):", quote=FALSE)
p <- anova(m.additiveCorefs, m.interactionCorefs.partial)
p

print("ANOVA comparison of fully interactive model and partial interactive model (to conclude there is no significant difference):", quote=FALSE)
p <- anova(m.interactionCorefs.full, m.interactionCorefs.partial)
p

#m.interactionCorefs.lme4 <- lme4::lmer(RR ~ Wgt * poly(Round, 2) * scale(Tokens) * Corefs + (1 + Wgt | Dyad), data = df, REML = FALSE)
#stargazer(m.interactionCorefs.lme4, digits = 5, no.space=TRUE, single.row=TRUE,report = "vcst*", label = "tab:coefficients:meanRR", star.cutoffs = c(0.05, 0.01, 0.001), table.placement = "htb", intercept.top = TRUE, intercept.bottom = FALSE, notes.label = "", omit.table.layout = "=lda")


