library(visreg)
library(lme4)


dat <- read.csv('dists_long_format.csv')

# remove reference
dat <- dat[dat$subject_id != "Reference",]


# check structure of the data.frame
str(dat)

dat$subject_id <- as.factor(dat$subject_id)
dat$label <- as.factor(dat$label)
dat$condition <- as.factor(dat$condition)
dat$group <- as.factor(dat$group)
dat$feature <- as.factor(dat$feature)
dat$dist.type <- as.factor(dat$dist.type)

# re-check structure of the data.frame
str(dat)


FEATURE = "mfcc_edd"
DIST_TYPE = "AVG"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # conditionSlovak (l3) -0.09068
summary(lme3)


FEATURE = "wave2vec2"
DIST_TYPE = "AVG"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # conditionSlovak (l3)  0.08805



FEATURE = "mfcc_edd"
DIST_TYPE = "DTW"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # conditionSlovak (l3) -0.08166


FEATURE = "wave2vec2"
DIST_TYPE = "DTW"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # conditionSlovak (l3)  0.09050





FEATURE = "mfcc_edd"
DIST_TYPE = "AVG"

cat("===========================================\n")
cat("WARNING: WITHOUT random intercept\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme2)

#Fixed effects:
#                     Estimate Std. Error t value
#(Intercept)           0.99655    0.03546  28.100
#conditionSlovak (l3) -0.09089    0.03391  -2.680
#groupmultilingual    -0.17387    0.03263  -5.329



# WITH norm

FEATURE = "mfcc_edd_ch_norm"
DIST_TYPE = "AVG"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme3)  # p < 0.01

# Fixed effects:
#                                        Estimate Std. Error t value
# (Intercept)                             1.10185    0.11045   9.976
# conditionSlovak (l3)                   -0.04533    0.06109  -0.742
# groupmultilingual                      -0.14872    0.12374  -1.202
# conditionSlovak (l3):groupmultilingual  0.08011    0.02958   2.708


FEATURE = "mfcc_edd_ch_norm"
DIST_TYPE = "DTW"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # p < 0.01  conditionSlovak (l3) -0.10581


FEATURE = "wave2vec2_ch_norm"
DIST_TYPE = "AVG"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # p < 0.001  conditionSlovak (l3)  0.08714


FEATURE = "wave2vec2_ch_norm"
DIST_TYPE = "DTW"

cat("===========================================\n")
cat(paste("feature: ", FEATURE, "dist type: ", DIST_TYPE, "\n"))
cat("===========================================\n")

dat_mfcc_avg <- dat[dat$feature == FEATURE & dat$dist.type == DIST_TYPE,]


lme0 <- lmer(dist ~ 1 + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.1 <- lmer(dist ~ condition + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme1.2 <- lmer(dist ~ group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme2 <- lmer(dist ~ condition + group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)
lme3 <- lmer(dist ~ condition * group + (1 | subject_id) + (1 | label), data=dat_mfcc_avg)

anova(lme0, lme1.1, lme2, lme3)
anova(lme0, lme1.2, lme2, lme3)
summary(lme1.1)  # p < 0.001  conditionSlovak (l3)  0.07106

