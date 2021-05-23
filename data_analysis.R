#################
##Libraries#####
################
library(plotrix)
library(reshape2)
library(dplyr)
library(ggplot2)
library(stringr)
library(RColorBrewer)
library("ggpubr")
library(tidyverse)
library(grid)
library(gridExtra)
coul <- brewer.pal(5, "Set2")
################

###Read CSV Data###
##################

data = read.csv("data.csv", stringsAsFactors = FALSE)

names(data)

######################


######################
#VA Expertise
########
myvars <- c('va_expertise')
va_expertise <- data[myvars]
va_expertise$va_expertise[va_expertise$va_expertise == 4] <- "Expert"
va_expertise$va_expertise[va_expertise$va_expertise == 3] <- "Knowledgeable"
va_expertise$va_expertise[va_expertise$va_expertise == 2] <- "Familiar"
va_expertise$va_expertise[va_expertise$va_expertise == 1] <- "Basic"
va_expertise_count <- table(va_expertise)

par(mar = rep(2, 4))
va_exp_plot<-barplot(va_expertise_count,
                 main="Visual Analytic Experience",
                 xlab="Expertise Range",
                 ylab="Count",
                 ylim=c(0,5),
                 col = coul)
text(va_exp_plot, va_expertise_count/2, paste(round(va_expertise_count/sum(va_expertise_count)*100,1), "%") ,cex=1)

######################
#Industry Expertise
########
myvars <- c('industry_expertise')
industry_expertise <- data[myvars]
industry_expertise$industry_expertise[industry_expertise$industry_expertise == 4] <- "First-hand"
industry_expertise$industry_expertise[industry_expertise$industry_expertise == 3] <- "Support"
industry_expertise$industry_expertise[industry_expertise$industry_expertise == 2] <- "Research experience"
industry_expertise$industry_expertise[industry_expertise$industry_expertise == 1] <- "No experience"
industry_expertise_count <- table(industry_expertise)

par(mar = rep(2, 4))
industry_plot<-barplot(industry_expertise_count,
                 main="Industry Experience",
                 xlab="Expertise Range",
                 ylab="Count",
                 ylim=c(0,5),
                 col = coul)
text(industry_plot, industry_expertise_count/2, paste(round(industry_expertise_count/sum(industry_expertise_count)*100,1), "%") ,cex=1)


######################
#Blow-molding Expertise
########
myvars <- c('blow_molding_expertise')
blow_molding_expertise <- data[myvars]
blow_molding_expertise$blow_molding_expertise[blow_molding_expertise$blow_molding_expertise == 4] <- "Expert"
blow_molding_expertise$blow_molding_expertise[blow_molding_expertise$blow_molding_expertise == 3] <- "Knowledgeable"
blow_molding_expertise$blow_molding_expertise[blow_molding_expertise$blow_molding_expertise == 2] <- "Familiar"
blow_molding_expertise$blow_molding_expertise[blow_molding_expertise$blow_molding_expertise == 1] <- "No Experience"
blow_molding_expertise_count <- table(blow_molding_expertise)

par(mar = rep(2, 4))
bm_plot<-barplot(blow_molding_expertise_count,
                 main="Experience with Blow Moulding Machines",
                 xlab="Expertise Range",
                 ylab="Count",
                 ylim=c(0,5),
                 col = coul)
text(bm_plot, blow_molding_expertise_count/2, paste(round(blow_molding_expertise_count/sum(blow_molding_expertise_count)*100,1), "%") ,cex=1)


######################
#VA Experience years
########
myvars <- c('va_years')
va_years <- data[myvars]
va_years$va_years[va_years$va_years == 4] <- "10 years or more"
va_years$va_years[va_years$va_years == 3] <- "5 to 10 years"
va_years$va_years[va_years$va_years == 2] <- "1 to 5 years"
va_years$va_years[va_years$va_years == 1] <- "0 years"
va_years_count <- table(va_years)

par(mar = rep(2, 4))

va_exp_plot<-barplot(va_years_count,
                 main="Years of VA Experience",
                 xlab="Years Range",
                 ylab="Count",
                 ylim=c(0,5),
                 col = coul)
text(va_exp_plot, va_years_count/2, paste(round(va_years_count/sum(va_years_count)*100,1), "%") ,cex=1)


par(mfrow=c(2,2))
va_exp_plot<-barplot(va_expertise_count,
                     main="Visual Analytic Experience",
                     xlab="Expertise Range",
                     ylab="Count",
                     ylim=c(0,5),
                     col = coul)
text(va_exp_plot, va_expertise_count/2, paste(round(va_expertise_count/sum(va_expertise_count)*100,1), "%") ,cex=1)

va_exp_plot<-barplot(va_years_count,
                     main="Years of VA Experience",
                     xlab="Years Range",
                     ylab="Count",
                     ylim=c(0,5),
                     col = coul)
text(va_exp_plot, va_years_count/2, paste(round(va_years_count/sum(va_years_count)*100,1), "%") ,cex=1)

industry_plot<-barplot(industry_expertise_count,
                       main="Industry Experience",
                       xlab="Expertise Range",
                       ylab="Count",
                       ylim=c(0,5),
                       col = coul)
text(industry_plot, industry_expertise_count/2, paste(round(industry_expertise_count/sum(industry_expertise_count)*100,1), "%") ,cex=1)

bm_plot<-barplot(blow_molding_expertise_count,
                 main="Experience with Blow Moulding Machines",
                 xlab="Expertise Range",
                 ylab="Count",
                 ylim=c(0,5),
                 col = coul)
text(bm_plot, blow_molding_expertise_count/2, paste(round(blow_molding_expertise_count/sum(blow_molding_expertise_count)*100,1), "%") ,cex=1)

######################
#ICE-T
####################
# 100 Insight - 3 > 8 (3-2-3)
###########
myvars <- c('icet111', 'icet112', 'icet113')
insight_1<- data[myvars]

insight_1 <- data.frame(insight_1$icet111, insight_1$icet112, insight_1$icet113, Average=rowMeans(insight_1))
i_var_answers <- summarize_all(insight_1, var, na.rm = TRUE)
i_average_answers <- summarize_all(insight_1, mean, na.rm = TRUE)
insight_1 <- rbind(insight_1, i_average_answers)
insight_1 <- rbind(insight_1, i_var_answers)

myvars <- c('icet121', 'icet122')
insight_2<- data[myvars]

insight_2 <- data.frame(insight_2$icet121, insight_2$icet122, Average=rowMeans(insight_2))
i_var_answers <- summarize_all(insight_2, var, na.rm = TRUE)
i_average_answers <- summarize_all(insight_2, mean, na.rm = TRUE)
insight_2 <- rbind(insight_2, i_average_answers)
insight_2 <- rbind(insight_2, i_var_answers)

myvars <- c('icet131', 'icet132', 'icet133')
insight_3<- data[myvars]

insight_3 <- data.frame(insight_3$icet131, insight_3$icet132, insight_3$icet133, Average=rowMeans(insight_3))
i_var_answers <- summarize_all(insight_3, var, na.rm = TRUE)
i_average_answers <- summarize_all(insight_3, mean, na.rm = TRUE)
insight_3 <- rbind(insight_3, i_average_answers)
insight_3 <- rbind(insight_3, i_var_answers)

myvars <- c('icet111', 'icet112', 'icet113', 'icet121', 'icet122', 'icet131', 'icet132', 'icet133')
insight<- data[myvars]

insight <- data.frame(insight$icet111, insight$icet112, insight$icet113, insight$icet121, insight$icet122, insight$icet131, insight$icet132, insight$icet133,Average=rowMeans(insight))
i_var_answers <- summarize_all(insight, var, na.rm = TRUE)
i_average_answers <- summarize_all(insight, mean, na.rm = TRUE)
insight <- rbind(insight, i_average_answers)
insight <- rbind(insight, i_var_answers)
##########
# 200 Time - 2 > 5 (2-3)
##########
myvars <- c('icet211', 'icet212')
time_1 <- data[myvars]
time_1 <- data.frame(time_1$icet211, time_1$icet212, Average=rowMeans(time_1))
t_var_answers <- summarize_all(time_1, var, na.rm = TRUE)
t_average_answers <- summarize_all(time_1, mean, na.rm = TRUE)
time_1 <- rbind(time_1, t_average_answers)
time_1 <- rbind(time_1, t_var_answers)

myvars <- c('icet221', 'icet222', 'icet223')
time_2 <- data[myvars]
time_2 <- data.frame(time_2$icet221, time_2$icet222, time_2$icet223,Average=rowMeans(time_2))
t_var_answers <- summarize_all(time_2, var, na.rm = TRUE)
t_average_answers <- summarize_all(time_2, mean, na.rm = TRUE)
time_2 <- rbind(time_2, t_average_answers)
time_2 <- rbind(time_2, t_var_answers)

myvars <- c('icet211', 'icet212', 'icet221', 'icet222', 'icet223')
time <- data[myvars]
time <- data.frame(time$icet211, time$icet212, time$icet221, time$icet222, time$icet223,Average=rowMeans(time))
t_var_answers <- summarize_all(time, var, na.rm = TRUE)
t_average_answers <- summarize_all(time, mean, na.rm = TRUE)
time <- rbind(time, t_average_answers)
time <- rbind(time, t_var_answers)
##############
# 300 Essence - 2 > 4 (2-2)
############
myvars <- c('icet311', 'icet312')
essence_1 <- data[myvars]
essence_1 <- data.frame(essence_1$icet311, essence_1$icet312, Average=rowMeans(essence_1))
e_var_answers <- summarize_all(essence_1, var, na.rm = TRUE)
e_average_answers <- summarize_all(essence_1, mean, na.rm = TRUE)
essence_1 <- rbind(essence_1, e_average_answers)
essence_1 <- rbind(essence_1, e_var_answers)

myvars <- c('icet321', 'icet322')
essence_2 <- data[myvars]
essence_2 <- data.frame(essence_2$icet321, essence_2$icet322, Average=rowMeans(essence_2))
e_var_answers <- summarize_all(essence_2, var, na.rm = TRUE)
e_average_answers <- summarize_all(essence_2, mean, na.rm = TRUE)
essence_2 <- rbind(essence_2, e_average_answers)
essence_2 <- rbind(essence_2, e_var_answers)

myvars <- c('icet311', 'icet312', 'icet321', 'icet322')
essence <- data[myvars]
essence <- data.frame(essence$icet311, essence$icet312, essence$icet321, essence$icet322, Average=rowMeans(essence))
e_var_answers <- summarize_all(essence, var, na.rm = TRUE)
e_average_answers <- summarize_all(essence, mean, na.rm = TRUE)
essence <- rbind(essence, e_average_answers)
essence <- rbind(essence, e_var_answers)

##############
# 400 Confidence - 3 > 4 (2-1)
################
myvars <- c('icet411', 'icet412')
confidence_1<- data[myvars]
confidence_1 <- data.frame(confidence_1$icet411, confidence_1$icet412, Average=round(rowMeans(confidence_1),1))
c_var_answers <- summarize_all(confidence_1, var, na.rm = TRUE)
c_average_answers <- summarize_all(confidence_1, mean, na.rm = TRUE)
confidence_1 <- rbind(confidence_1, c_average_answers)
confidence_1 <- rbind(confidence_1, c_var_answers)

myvars <- c('icet421')
confidence_2 <- data[myvars]
confidence_2 <- data.frame(confidence_2$icet421, Average=round(rowMeans(confidence_2),1))
c_var_answers <- summarize_all(confidence_2, var, na.rm = TRUE)
c_average_answers <- summarize_all(confidence_2, mean, na.rm = TRUE)
confidence_2 <- rbind(confidence_2, c_average_answers)
confidence_2 <- rbind(confidence_2, c_var_answers)

myvars <- c('icet411', 'icet412', 'icet421')
confidence <- data[myvars]
confidence <- data.frame(confidence$icet411, confidence$icet412, confidence$icet421, Average=round(rowMeans(confidence),1))
c_var_answers <- summarize_all(confidence, var, na.rm = TRUE)
c_average_answers <- summarize_all(confidence, mean, na.rm = TRUE)
confidence <- rbind(confidence, c_average_answers)
confidence <- rbind(confidence, c_var_answers)

#################

# Table summary(Add color coding)
#################
summary_table <- data.frame(insight_1$Average, insight_2$Average, insight_3$Average, time_1$Average, time_2$Average,essence_1$Average, essence_2$Average, confidence_1$Average, confidence_2$Average)
summary_table <- data.frame(insight_1$Average, insight_2$Average, insight_3$Average, time_1$Average, time_2$Average,essence_1$Average, essence_2$Average, confidence_1$Average, confidence_2$Average, Average=round(rowMeans(summary_table),2))

theinsight <- data.frame(insight_1$Average, insight_2$Average, insight_3$Average)
theinsight <- data.frame(insight_1$Average, insight_2$Average, insight_3$Average, Average=round(rowMeans(theinsight),2))

thetime <- data.frame(time_1$Average, time_2$Average)
thetime <- data.frame(time_1$Average, time_2$Average, Average=round(rowMeans(thetime),2))

theessence <- data.frame(essence_1$Average, essence_2$Average)
theessence <- data.frame(essence_1$Average, essence_2$Average, Average=round(rowMeans(theessence),2))

theconfidence <- data.frame(confidence_1$Average, confidence_2$Average)
theconfidence <- data.frame(confidence_1$Average, confidence_2$Average, Average=round(rowMeans(theconfidence),2))

table_scores_each <- data.frame(theinsight$Average, thetime$Average, theessence$Average, theconfidence$Average)
table_scores_each <- data.frame(theinsight$Average, thetime$Average, theessence$Average, theconfidence$Average, Average=round(rowMeans(table_scores_each),2))
table_scores_each  <- table_scores_each[1:6,]

colnames(table_scores_each) <- c('   Insight (I)', '   Time (T)', '  Essence (E)', ' Confidence (C)', 'Average')
rownames(table_scores_each) <- c('X1', ' X2', '  X3', '   X4', '     X5', '     Average')



heatmap_df <- table_scores_each %>%
        rownames_to_column() %>%
        gather(colname, value, -rowname)

rng = range(1,7)
ggplot(heatmap_df, aes(x = colname, y = rowname, fill = value)) +
        geom_tile() +
        geom_text(aes(label = format(round(value, 2), nsmall = 2) )) +
        scale_fill_gradient2(low = "#dc3b3b",
                            high = "#31a354",
                            guide = "colorbar",  
                            midpoint=mean(rng), 
                            breaks=seq(1,7,1),
                            limits=c(1, 7))+
        theme(axis.title.x=element_blank(), 
              axis.title.y=element_blank(), 
              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank())+ 
        scale_x_discrete(position = "top") 

high_level <- c(
        'Insight',
        'Insight',
        'Insight',
        'Insight',
        'Insight',
        'Insight',
        'Insight',
        'Insight',
        'Time',
        'Time',
        'Time',
        'Time',
        'Time',
        'Essence',
        'Essence',
        'Essence',
        'Essence',
        'Confidence',
        'Confidence',
        'Confidence')

mid_level <- c(
        '1.The visualization facilitates answering questions about the data',
        '1.The visualization facilitates answering questions about the data',
        '1.The visualization facilitates answering questions about the data',
        '2.The visualization provides a new or better understanding of the data',
        '2.The visualization provides a new or better understanding of the data',
        '3.The visualization provides opportunities for serendipitous discoveries',
        '3.The visualization provides opportunities for serendipitous discoveries',
        '3.The visualization provides opportunities for serendipitous discoveries',
        '4.The visualization affords rapid parallel comprehension for efficient browsing',
        '4.The visualization affords rapid parallel comprehension for efficient browsing',
        '5.The visualization provides mechanisms for quickly seeking specific information',
        '5.The visualization provides mechanisms for quickly seeking specific information',
        '5.The visualization provides mechanisms for quickly seeking specific information',
        '6.The visualization provides a big picture perspective of the data',
        '6.The visualization provides a big picture perspective of the data',
        '7.The visualization provides an understanding of the data beyond individual data cases',
        '7.The visualization provides an understanding of the data beyond individual data cases',
        '8.The visualization helps avoid making incorrect inferences',
        '8.The visualization helps avoid making incorrect inferences',
        '9.The visualization facilitates learning more broadly about the domain of the data')

questions <- c(
        'The visualization exposes individual data cases and their attributes', 
        'The visualization facilitates perceiving relationships in the data like patterns & distributions of the variables', 
        'The visualization promotes exploring relationships between individual data cases as well as different groupings of data cases',
        'The visualization helps generate data-driven questions',
        'The visualization helps identify unusual or unexpected, yet valid, data characteristics or values',
        'The visualization provides useful interactive capabilities to help investigate the data in multiple ways',
        'The visualization shows multiple perspectives about the data',
        'The visualization uses an effective representation of the data that shows related and partially related data cases',
        'The visualization provides a meaningful spatial organization of the data',
        'The visualization shows key characteristics of the data at a glance',
        'The interface supports using different attributes of the data to reorganize the visualization\'s appearance',
        'The visualization supports smooth transitions between different levels of detail in viewing the data',
        'The visualization avoids complex commands and textual queries by providing direct interaction with the data representation',
        'The visualization provides a comprehensive and accessible overview of the data',
        'he visualization presents the data by providing a meaningful visual schema',
        'The visualization facilitates generalizations and extrapolations of patterns and conclusions',
        'The visualization helps understand how variables relate in order to accomplish different analytic tasks',
        'The visualization uses meaningful and accurate visual encodings to represent the data',
        'The visualization avoids using misleading representations',
        'The visualization promotes understanding data domain characteristics beyond the individual data cases and attributes')
averages <- c(as.numeric(i_average_answers[1:8]), as.numeric(t_average_answers[1:5]), as.numeric(e_average_answers[1:4]), as.numeric(c_average_answers[1:3]))
#variances <- c(as.numeric(i_var_answers[1:8]), as.numeric(t_var_answers[1:5]), as.numeric(e_var_answers[1:4]), as.numeric(c_var_answers[1:3]))


#entire_evaluation_Table <- data.frame(high_level, mid_level, questions, averages, variances)
entire_evaluation_Table <- data.frame(high_level, mid_level, questions, averages)


mid_level_table <-  entire_evaluation_Table %>%
        group_by(mid_level) %>%
        summarise(avg = mean(averages))

#mid_level_table <- cbind(higher_level, mid_level_table)
mid_level_table_ <- mid_level_table
keep <- c("avg")
mid_level_table_<-mid_level_table_[keep]
colnames(mid_level_table_) <- c('Average')
rownames(mid_level_table_) <- c('1.[I] The visualization facilitates answering questions about the data',
                                '2.[I] The visualization provides a new or better understanding of the data',
                                '3.[I] The visualization provides opportunities for serendipitous discoveries',
                                '4.[T] The visualization affords rapid parallel comprehension for efficient browsing',
                                '5.[T] The visualization provides mechanisms for quickly seeking specific information',
                                '6.[E] The visualization provides a big picture perspective of the data',
                                '7.[E] The visualization provides an understanding of the data beyond individual data cases',
                                '8.[C] The visualization helps avoid making incorrect inferences',
                                '9.[C] The visualization facilitates learning more broadly about the domain of the data')

heatmap_ml <- mid_level_table_ %>%
        rownames_to_column() %>%
        gather(colname, value, -rowname)

rng = range(1,7)
ggplot(heatmap_ml, aes(x = colname, y = rowname, fill = value)) +
        geom_tile() +
        geom_text(aes(label = format(round(value, 2), nsmall = 2) )) +
        scale_fill_gradient2(low = "#dc3b3b",
                             high = "#31a354",
                             guide = "colorbar",  
                             midpoint=mean(rng), 
                             breaks=seq(1,7,1),
                             limits=c(1, 7))+
        theme(axis.title.x=element_blank(), 
              axis.title.y=element_blank(), 
              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.y = element_text(hjust = 0))+ 
        scale_x_discrete(position = "top") +
        scale_y_discrete(limits=rev)

insight_bp <- data.frame(data$icet111, data$icet112, data$icet113, 
                         data$icet121, data$icet122, 
                         data$icet131, data$icet132, data$icet133)
time_bp <- data.frame(data$icet211, data$icet212, data$icet221, data$icet222, data$icet223)
essence_bp <- data.frame(data$icet311, data$icet312, data$icet321, data$icet322)
confidence_bp <- data.frame(data$icet411, data$icet412, data$icet421)
colnames(insight_bp) <- c('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8')
colnames(time_bp) <- c('S9', 'S10', 'S11', 'S12', 'S13')
colnames(essence_bp) <- c('S14', 'S15', 'S16', 'S17')
colnames(confidence_bp ) <- c('S18', 'S19', 'S20')


boxplot(insight_bp[,1:8], data=insight_bp, main="Insight",
        xlab="Criteria", col = coul, ylab="Response", ylim=c(4,7))

boxplot(time_bp[,1:5], data=time_bp, main="Time",
        xlab="Criteria", col = coul, ylab="Response", ylim=c(4,7))

boxplot(essence_bp[,1:4], data=essence_bp, main="Essence",
        xlab="Criteria", col = coul, ylab="Response", ylim=c(4,7))

boxplot(confidence_bp[,1:3], data=confidence_bp, main="Confidence",
        xlab="Criteria", col = coul, ylab="Response", ylim=c(4,7))

index_stat <- c('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
                'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 
                'S18', 'S19', 'S20')
index_statements <- data.frame(index_stat, questions)
colnames(index_statements) <- c('Index', 'Statements')

png("statements_index.png", height = 30*nrow(index_statements), width = 500*ncol(index_statements))
grid.table(index_statements)
dev.off()

######################
#SUS
####################
# SUS - 10
vectors <- c('sus0', 'sus1', 'sus2', 'sus3', 'sus4', 'sus5', 'sus6', 'sus7', 'sus8', 'sus9')
sus <- data[vectors]

sus$sus1[sus$sus1==4] <-'zero'
sus$sus1[sus$sus1==3] <-'one'
sus$sus1[sus$sus1==1] <-3
sus$sus1[sus$sus1==0] <-4
sus$sus1[sus$sus1=='zero'] <- 0
sus$sus1[sus$sus1=='one'] <- 1

sus$sus3[sus$sus3==4] <-'zero'
sus$sus3[sus$sus3==3] <-'one'
sus$sus3[sus$sus3==1] <-3
sus$sus3[sus$sus3==0] <-4
sus$sus3[sus$sus3=='zero'] <- 0
sus$sus3[sus$sus3=='one'] <- 1

sus$sus5[sus$sus5==4] <-'zero'
sus$sus5[sus$sus5==3] <-'one'
sus$sus5[sus$sus5==1] <-3
sus$sus5[sus$sus5==0] <-4
sus$sus5[sus$sus5=='zero'] <- 0
sus$sus5[sus$sus5=='one'] <- 1

sus$sus7[sus$sus7==4] <-'zero'
sus$sus7[sus$sus7==3] <-'one'
sus$sus7[sus$sus7==1] <-3
sus$sus7[sus$sus7==0] <-4
sus$sus7[sus$sus7=='zero'] <- 0
sus$sus7[sus$sus7=='one'] <- 1

sus$sus9[sus$sus9==4] <-'zero'
sus$sus9[sus$sus9==3] <-'one'
sus$sus9[sus$sus9==1] <-3
sus$sus9[sus$sus9==0] <-4
sus$sus9[sus$sus9=='zero'] <- 0
sus$sus9[sus$sus9=='one'] <- 1

sus$sus1 <- as.numeric(sus$sus1)
sus$sus3 <- as.numeric(sus$sus3)
sus$sus5 <- as.numeric(sus$sus5)
sus$sus7 <- as.numeric(sus$sus7)
sus$sus9 <- as.numeric(sus$sus9)

#Based on research, a SUS score above a 68 would be considered above average and 
#anything below 68 is below average, 
#however the best way to interpret your results involves “normalizing” the scores to produce a percentile ranking.

sus <- data.frame(sus$sus0, sus$sus1, sus$sus2, sus$sus3, sus$sus4, 
                   sus$sus5, sus$sus6, sus$sus7, sus$sus8, sus$sus9, Percentile=(rowSums(sus)*2.5))

rownames(sus) <- c('X1', ' X2', '  X3', '   X4', '     X5')

colnames(sus) <- c('SUS1', 'SUS2', 'SUS3', 'SUS4', 'SUS5', 'SUS6', 'SUS7', 'SUS8', 'SUS9', 'SUS10', 'Percentile')

sus_ <- sus
sus_$Percentile <- sus_$Percentile/25
colnames(sus_) <- c('  SUS1', '  SUS2', '  SUS3', '  SUS4', '  SUS5', '  SUS6', '  SUS7', '  SUS8', '  SUS9', ' SUS10', 'Average')
averages_sus <- c(mean(sus_$`  SUS1`), mean(sus_$`  SUS2`), mean(sus_$`  SUS3`), mean(sus_$`  SUS4`),
                  mean(sus_$`  SUS5`), mean(sus_$`  SUS6`), mean(sus_$`  SUS7`), mean(sus_$`  SUS8`), 
                  mean(sus_$`  SUS9`), mean(sus_$` SUS10`), mean(sus_$Average))
sus_ <- rbind(sus_, averages_sus)
rownames(sus_) <- c('X1', ' X2', '  X3', '   X4', '     X5', '     Averages')
heatmap_sus <- sus_ %>%
        rownames_to_column() %>%
        gather(colname, value, -rowname)

rng = range(0,4)
ggplot(heatmap_sus, aes(x = colname, y = rowname, fill = value)) +
        geom_tile() +
        geom_text(aes(label = format(round(value, 2), nsmall = 2) )) +
        scale_fill_gradient2(low = "#dc3b3b",
                             high = "#31a354",
                             guide = "colorbar",  
                             midpoint=2.72, 
                             breaks=seq(0,4,0.5),
                             limits=c(0, 4))+
        theme(axis.title.x=element_blank(), 
              axis.title.y=element_blank(), 
              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank())+ 
        scale_x_discrete(position = "top") 

par(mar = c(3, 3, 3, 3))
par(mfrow=c(1,1))
boxplot(sus[,1:10], data=sus, main="System Usability",
        xlab="Criteria", col = coul, ylab="Response", ylim=c(0,4))
