# Goals of this Script
#
# Calculate task-wise FD averages / task / subject
# Ian Richard Ferguson | Stanford University

sink("/dev/null")

# ------ Imports
library(dplyr)
library(ggplot2)
library(stringr)

args = commandArgs(trailingOnly=T)                                              # Pass in ./group_bold.tsv at command line
group_bold = read.csv(args[1], sep="\t")                                        # Read as DataFrame object

reduced <- group_bold %>% 
        select(bids_name, contains("fd_")) %>%                                  # Select variables of interest (framewise displacement metrics)
        mutate(subjID = str_split(bids_name, "_", simplify = T)[,1],            # Isolate subject ID
               task = str_split(bids_name, "_", simplify = T)[,2])              # Isolate task ID

reduced %>% 
        group_by(subjID, task) %>% 
        summarise(run.count = n(),                                              # Number of instances / task / sub
                  avg.fd.pct = mean(fd_perc, na.rm=T)) %>%                      # Average framewise displacement / task
        write.csv(file="./taskwise-summary.csv")                                # Write to local CSV
