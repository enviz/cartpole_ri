import tensorflow as tf, Game, numpy as np, os, random
from collections import deque
                                                      
game = Game.Game()                                    #instantiate the environment cartpole form openai gym
#game.env._max_episode_steps = 10000                   #source : https://gym.openai.com/docs/
model = []
retrain = True
state_size = 4              #states: pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
action_size = 2               #action can be either left or right

if not os.path.exists('new_models/model') or retrain:
    print("new model")
    if not os.path.exists('new_models/'):
        os.mkdir('new_models')                            ###define our model
    inputs = tf.keras.Input(shape=(state_size,))          

    layer1 = tf.keras.layers.Dense(24, activation=tf.keras.activations.relu)(inputs)
    layer2 = tf.keras.layers.Dense(24, activation=tf.keras.activations.relu)(layer1)

    output = tf.keras.layers.Dense(action_size, activation=tf.keras.activations.linear)(layer2)     

    model = tf.keras.Model(inputs=inputs, outputs = output)

    optimizer = tf.keras.optimizers.Adam(1e-3)


    model.compile(loss=tf.keras.losses.mean_squared_error.__name__, optimizer = optimizer)

    num_episodes = 5000               
    
    ## Our goal is to keep the pole upright as long as possible(which is our reward)
   

    epsilon = 1.0          #eps is the exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
    epsilon_min = 0.25     #we want the agent to explore at least this amount
    epsilon_decay = 0.995   #we want to decrease the number of explorations as it gets good at playing

    memory = deque(maxlen=2000)          

    gamma = 0.95
##
##    if retrain:
##        model = tf.keras.models.load_model('new_models/model')                           

    for episode in range(num_episodes):
        cur_state = game.env.reset()        #reset the game to initial state. 
                                            
        cur_state = np.reshape(cur_state, [1, 4])      #reshaping it to give it to our model
        time_t = 0                  # time_t represents each frame of the game,which is also indicative of how much you have been keeping the pole upright.
        ## Our goal is to keep the pole upright as long as possible. 
        #So,the score of an episode is the number of frames played,which is time_t
        while True:
            time_t += 1          
            # predict what action to take 
            rewards = model.predict(cur_state)           #predicted reward for each action
            best_action = 0
            #agent decides to act randomly first
            if np.random.rand() >= epsilon:
                best_action = game.env.action_space.sample()      #chooses a random action
            else:
                best_action = np.argmax(rewards[0])   #rewards[0] contains the rewards associated with each action
                                                      #so we choose that action that has the maximum predicted reward  

            next_state, reward, done, info = game.env.step(best_action)        #find next_state and actual reward after taking the action that the model predicted
            next_state = np.reshape(next_state, [1, 4])                      
            memory.append((cur_state, best_action, next_state, reward, done))      #recording each state,action,reward,next_state in  memory 
            cur_state = next_state                                                #make  current_state as the next_state for the next turn
            if done:                                                                  #done is  a boolean value True/False which indicates whether the episode has finished/terminated.
                print("Episode:", episode, " Score:", time_t, " Epsilon:", epsilon)   #printing the score after each episode.
                break
        if len(memory) >= 32:                                         
                                                            # train the agent with the experience of previous episodes. Here we are giving a batch size of 32
            sample = random.sample(memory, 32)              #random sample of 32 episodes to learn from memory.
            for cur_state, best_action, next_state, reward, done in sample:
                potential_reward = reward
                if not done:
                    potential_reward = reward + gamma*np.amax(model.predict(next_state)[0])   #To make the agent perform well in long-term, 
                    #immediate_reward + discounted future rewards                           #we take into account not only the immediate rewards 
                     #gamma can be between (0,1)                                                                   
                                                                        #we are going to give a ‘discount rate’ or ‘gamma’.
                # make the agent to approximately map
                # the current state to future discounted reward                                                                          
                target_reward = model.predict(cur_state)                       #predict rewards for a state and update the reward associated for each action
                target_reward[0][best_action] = potential_reward               #update and store it in target_reward
                model.fit(cur_state,target_reward, epochs=1,verbose=0)         #train model using current_state and target_reward for 1 epoch.
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            else:
                epsilon = 1.0
    tf.keras.models.save_model(model, 'new_models/model')            
else:
    model = tf.keras.models.load_model('new_models/model')   
cur_state = game.env.reset()                    #now that we have learnt how to play,let's see our agent in action
                                                #now we don't take random actions
cur_state = np.reshape(cur_state, [1, 4])
game_over = False
time = 0
while not game_over:                               
    game.env.render()                             #render to see the output of the cartpole
    time += 1
    action = np.argmax(model.predict(cur_state))          #predict action
    observation, reward, done, info = game.env.step(action)     #find reward for taking the action
    cur_state = np.reshape(observation, [1, 4])
    game_over = done
print(time)
game.env.close()
