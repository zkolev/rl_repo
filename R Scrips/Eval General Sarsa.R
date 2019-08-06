## SETUP: 


## LOAD LIBRARIES: 
require(tidyverse)
require(reticulate)

## SET WD: 
setwd('C:\\006 Learning\\RL')

## SETUP PYTHON ENV:
reticulate::use_condaenv('dlr_env')


## DEFINE UTILITY FUNCTIONS: 


### THis function takes the policy object pickled from 
### the python environment and structures it as a data frame 

structure_policy <- function(policy_list, num_checkouts = 2) {
  
  library(foreach)
  
  ## Loop through each state
  ## and obtain each action for the state: 
  
  
  policy_df = foreach(i = seq_along(policy_list))  %do% {
    
    ## The names of the list elements contains the 
    ## information about the state of the game 
    policy_state <- names(policy_list[i])
    
    ## Separate each element with ','
    policy_state_elements <- strsplit(policy_state, ',')[[1]] 
    
    ## Clean  the policy element vectors
    policy_state_elements_cleaned <- policy_state_elements[!grepl('\\|',policy_state_elements)] ## remove the separator elements '|' 
    policy_state_elements_cleaned <- gsub("[^[:alnum:]]" , '', policy_state_elements_cleaned)
    
    ## Obtain the action:
    policy_action = policy_list[[i]]
    
    ## There are two possible actions:
    ## Roll and Checkout
    
    
    if(policy_action[[1]] == 'Roll') {
      
      policy_action_elements = c(unlist(policy_action[[2]]), 'Roll')
      
    
      } else if(policy_action[[1]] == 'TerminalState') {
        
        policy_action_elements = c(rep(0,6), 'TerminalState')
        
      } else {
      
        policy_action_elements = c(rep(0,6), policy_action[[2]])
    }
    
    policy_df_vector = c(policy_state, policy_state_elements_cleaned, policy_action_elements)
    
    return(policy_df_vector)
    
  }

  ## convert to matrix
  
  policy_df = do.call(rbind, policy_df)
  policy_df = tibble::as.tibble(policy_df)
  # names(policy_df) <- 

  return(policy_df)
  
  
}


## This function takes the pickled action value nested dict 
## and structures it as data frame object

structure_av <- function(my_avalue) {
  
  foreach(i = seq_along(my_avalue)) %do% {
    game_state <- names(my_avalue[i])
    foreach(j = seq_along(my_avalue[[i]])) %do% {
      
      action <- names(my_avalue[[i]][j])
      value  <- my_avalue[[i]][[j]]
      
    }
    
    structured_df = tibble(state = game_state
                           ,action = action
                           ,value = value)
    
    return(structured_df)
    
  }
  
  
}



## LOAD DATA: 
my_policy = py_load_object("C:\\006 Learning\\RL\\SARSA RESULTS\\policy.pickle")
my_avalue = py_load_object("C:\\006 Learning\\RL\\SARSA RESULTS\\action_value.pickle")


baseline_reward <- py_load_object("C:\\006 Learning\\RL\\SARSA RESULTS\\baseline_reward.pickle")
sarsa_reward <- py_load_object("C:\\006 Learning\\RL\\SARSA RESULTS\\training_rewards.pickle")
eval_rew = py_load_object("C:\\006 Learning\\RL\\SARSA RESULTS\\evaluated_rewards.pickle")
  

## Structure policy ## 

policy_structured <- structure_policy(my_policy)


## Names of the structured policy 
names_vector <- c('key','s_remaining_rolls'
                 , paste('s_f', 1:6, sep = '_')
                 , paste('s_co', c('ones', 'twos'), sep = '_')
                 , paste('a',1:6,'roll', sep ='_')
                 , 'a_Ck_out'
                  )

## Rename the policy
names(policy_structured) <- names_vector


# ## Structure action value ## 
# avalue_structured <- structure_av(my_avalue = my_avalue)
# avalue_structured <- do.call(bind_rows, avalue_structured)

##
baseline_df = tibble::as_tibble(do.call(rbind, lapply(baseline_reward,c))) %>% unnest()
evaluated_df = tibble::as_tibble(do.call(rbind, lapply(sarsa_reward,c))) %>% unnest()
eval_samples = tibble::as_tibble(do.call(cbind, lapply(eval_rew,c))) %>% unnest()
  

## Aggregate the baseline function:
baseline_shares = baseline_df %>% 
  group_by(V2) %>%
  summarise(tot = n()) %>%
  ungroup() %>% 
  transmute(tot = tot/sum(tot)
        ,field = -1
        ,value = V2 
        )

## Inspect the model performance 
## As it evolves

## Each bar represents an set of played games
## after N rounds of policy iteration

eval_samples %>% 
  gather(field, value)  %>%
  group_by(field, value) %>% 
  summarise(tot = n()) %>%
  ungroup()  %>%
  group_by(field) %>% 
  mutate(tot = tot/sum(tot)) %>% 
  ungroup() %>% 
  mutate(field = as.numeric(gsub('V', '',field))) %>% 
  bind_rows(baseline_shares) %>% 
  ggplot(aes(x = factor(field), y = tot, fill = factor(value))) + 
  geom_bar(stat = 'identity', position = 'fill')





mutate(baseline_df, fl = 'base') %>%
  bind_rows(mutate(evaluated_df, fl = 'base2') %>% slice(20000:n())) %>%
  group_by(V2, fl) %>% 
  summarise(tot = n()) %>% 
  group_by(fl) %>% 
  mutate(tot = tot/sum(tot)) %>%
  ggplot(aes(x= factor(V2), y = tot, fill = fl)) +
  geom_bar(stat = 'identity', position = 'dodge') + 
  geom_smooth()

### 

evaluated_df %>% 
  mutate(time_ix = 1:n()) %>%
  ggplot(aes(x = time_ix, y = V2))  + 
  geom_line() + 
  geom_smooth()

sarsa_reward
