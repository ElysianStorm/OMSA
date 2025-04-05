library(AER)

mydata <- read.csv("../datasets/MGT6203_HW5_2_Education_data.csv")
str(mydata)

hist(mydata$wage, breaks=50)
hist(log(mydata$wage), breaks=50)

ols.res <- lm(log(wage) ~ educ + exper + I(exper^2), data=mydata)
summary(ols.res)

stage1 <- lm(educ ~ nearc4 + exper + I(exper^2), data=mydata)
summary(stage1)

stage2 <- lm(log(wage) ~ fitted(stage1) + exper + I(exper^2), data=mydata)
summary(stage2)

TSLS.res <- ivreg(log(wage) ~ educ + exper + I(exper^2) | nearc4 + exper + I(exper^2), data=mydata)
summary(TSLS.res)

cbind(coef(ols.res), coef(TSLS.res))

