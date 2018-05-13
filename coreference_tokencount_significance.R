#!/usr/bin/env Rscript

# Calculates significance of effects of token count, round and coreference count on word referring ability (RA).
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
# Only select the rows using just "Wgt"
df <- df[df$cond %in% c("Wgt"), ]
# Hack to change legend label
names(df)[names(df) == "cond"] <- "Condition"
names(df)[names(df) == "rank"] <- "Rank"
names(df)[names(df) == "round"] <- "Round"
names(df)[names(df) == "session"] <- "Dyad"
names(df)[names(df) == "weight"] <- "MeanRA"
names(df)[names(df) == "words"] <- "Tokens"

df$RR <- 1.0 / df$Rank
df$Condition <- reorder(df$Condition, df$RR, FUN=mean)

# https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
control <- lmerControl(optimizer ="Nelder_Mead")
# Limited-memory BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS> in order to avoid failed convergence, caused by the addition of the "Tokens" condition
# See https://stats.stackexchange.com/a/243225/169001
#control <- lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B'))


print("Testing significance of relationship of \"RA\" with \"Tokens\", \"Round\" and \"Corefs\":", quote=FALSE)
m.raAdditive <- lmer(scale(Tokens) ~ Corefs + poly(Round, 2) + (1  | Dyad), data = df, REML = FALSE, control = control)
summary(m.raAdditive)

print("Fully interactive model:", quote=FALSE)
m.raInteractive.full <- lmer(scale(Tokens) ~ Corefs * poly(Round, 2) + (1  | Dyad), data = df, REML = FALSE, control = control)
summary(m.raInteractive.full)

p <- anova(m.raAdditive, m.raInteractive.full)
p
