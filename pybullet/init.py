ADD THE FOLLOWING INSTRICTIONS TO ....../pybullet_envs/__init__.py


register(id='AntBulletEnv-v5',  # reward optimized for evolutionary strategies
         entry_point='pybullet_envs.gym_locomotion_envs2:AntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='Walker2DBulletEnv-v5',
         entry_point='pybullet_envs.gym_locomotion_envs2:Walker2DBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HalfCheetahBulletEnv-v5',
         entry_point='pybullet_envs.gym_locomotion_envs2:HalfCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

register(id='AntBulletEnv-v5',
         entry_point='pybullet_envs.gym_locomotion_envs2:AntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HopperBulletEnv-v5',
         entry_point='pybullet_envs.gym_locomotion_envs2:HopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HumanoidBulletEnv-v5',
         entry_point='pybullet_envs.gym_locomotion_envs2:HumanoidBulletEnv',
         max_episode_steps=1000)
