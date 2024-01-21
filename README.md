# Investigating Plasticity in RL
By: Verona Teo*, Arvind Rajaraman*, and Seyone Chithrananda*

\* Equal contribution and co-authorship

This repo explores the problem of plasticity loss in deep reinforcement learning. **Plasticity loss** is the phenomenon of neural networks losing their ability to classify/regress on new data, when trained on old data until convergence. This term is derived from neuroplasticity loss, which is the loss of ability of human brains adapting to new experiences over time (which explains why it's harder for adults to learn a language as compared to children).


<table border="0" align="center">
  <tr>
    <td>
      <img width="400" alt="image" src="https://github.com/arvindrajaraman/plasticity-rl/assets/30243515/4a313ebd-0636-4002-b4ce-68e4b64e5e3a">
    </td>
    <td>
      <img width="400" alt="image" src="https://github.com/arvindrajaraman/plasticity-rl/assets/30243515/a6b4a20e-c546-4725-84f1-193e61f5c84f">
    </td>
  </tr>
</table>


Specifically, we investigate the loss of plasticity in deep Q-networks and analyze how the weights, rank, and activation neurons change over time in several experiments. We explore:
- Successive regularization for later layers in a neural network
- Regularizing singular values to create desirable convergence properties
- Selectively regularizing only during certain time steps
- Regularizing weights towards the distribution of initial weights, not just towards 0 weight magnitude

This project was done as a part of CS 285 (Deep Reinforcement Learning) at UC Berkeley in the Fall 2023 semester.
