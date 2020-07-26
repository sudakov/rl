import ray
import ray.rllib.agents.ppo as ppo
#from ray.tune.logger import pretty_print
# rllib train -f cartpole-ppo.yaml

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
# config["num_gpus"] = 0
# config["num_workers"] = 1
# trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.
"""
for i in range(3):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
"""
agent = ppo.PPOTrainer(env="CartPole-v0", config={
        'framework': 'tf',
        'gamma': 0.99,
        'lr': 0.0003,
        'num_workers': 1,
        'observation_filter': 'MeanStdFilter',
        'num_sgd_iter': 6,
        'vf_share_layers': 'true',
        'vf_loss_coeff': 0.01,
        'model':
          {'fcnet_hiddens': [32],
           'fcnet_activation': 'linear'
          }
} )
agent.restore("checkpoint-7")

import gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 5000
for i_episode in range(20):
    observation = env.reset()
    for t in range(800):
        env.render()
        #print(observation)
        action = agent.compute_action(observation)
        # print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
env.close()
