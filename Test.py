import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from QLearn import Catch

# true35tensorflowgpu    environment runs it and trains it pretty fast
if __name__ == "__main__":
    # Make sure this grid size matches the value used for training
    grid_size = 10
    model =  tf.keras.models.load_model("model.h5")
    #model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(10):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()
        plt.imshow(input_t.reshape((grid_size,)*2),
                   interpolation='none', cmap='gray')
        #plt.show()
        #plt.savefig("%03d.png" % c)
        c += 1
        reward=0
        while not game_over:
            input_tm1 = input_t
            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])
            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            plt.imshow(input_t.reshape((grid_size,)*2),
                       interpolation='none', cmap='gray')
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            #plt.savefig("%03d.png" % c)
            c += 1
        if reward == 1:
            print('game won---')
        else:
            print('reward = ' + str(reward))
