
# RTify: Aligning Deep Neural Networks with Human Behavioral Decisions


_Yu-Ang Cheng, Ivan Felipe Rodriguez, Sixuan Chen, Kohitij Kar, Takeo Watanabe, Thomas Serre_

<p align="center">
<a href="https://arxiv.org/pdf/2411.03630"><strong>Read the official paper »</strong></a>
</p>

## Paper Summary

 We present RTify, a novel computational approach to optimize the recurrent steps of RNNs to account for human RTs. 
 With this framework, we successfully fit an RNN directly to  human behavioral responses.​
 Our framework can also be extended to an ideal-observer model whereby the RNN is trained without human data via a penalty term that encourages the network to make a decision as quickly as possible.​ Under this setting, human-like behavioral responses naturally emerge from the RNN.​


## Contribution

- **Dynamic Evidence Accumulation**: Learnable evidence function that accumulates over time and triggers decision-making at optimal points, reflecting human-like RTs.
- **Human RT Fitting**: Supervised training mode where the model is aligned to observed human reaction times.
- **Self-Penalized Training**: An unsupervised mode promoting the model's speed-accuracy trade-off through a custom regularization term, emulating human-like decision timing without explicit RT data.

## Architecture

RTify can be incorporated into any pre-trained RNN model, for example, Wong-Wang (WW) model or convolutional network. The core process involves:
- Using a function \( f_w \) to convert RNN hidden states \( h_t \) into an evidence score \( e_t \).
- Accumulating evidence \( Φ_t \) over time until it surpasses a learnable threshold \( θ \), at which point the model outputs a decision.
- Recording the time step \( τ_θ \), where the evidence threshold is first crossed as the reaction time.


![Architecture Diagram](figures/Fig1.jpg)


[//]: # (## Results)

[//]: # ()
[//]: # (- **RDM Task**: RTify models accurately capture the distribution of human RTs across varying coherence levels, outperforming entropy-thresholding baselines.)

[//]: # (![Results Diagram]&#40;figures/Fig2.jpg&#41;)

[//]: # (- **Object Recognition**: RTify aligns with human RTs in image classification tasks, showcasing robust performance even in complex visual tasks.)

[//]: # (![Results Diagram]&#40;figures/Fig5.jpg&#41;)


## Citation

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper](https://arxiv.org/pdf/2411.03630):

```
@article{cheng2024rtify,
  title={RTify: Aligning Deep Neural Networks with Human Behavioral Decisions},
  author={Cheng, Yu-Ang and Rodriguez, Ivan Felipe and Chen, Sixuan and Kar, Kohitij and Watanabe, Takeo and Serre, Thomas},
  journal={arXiv preprint arXiv:2411.03630},
  year={2024}
}
```


## Acknowledgments

This work was supported by NSF (IIS-2402875), ONR (N00014-24-1-2026) and the ANR-3IA Artificial and Natural Intelligence Toulouse Institute (ANR-19-PI3A-0004) to T.S and National Institutes of Health (NIH R01EY019466 and R01EY027841) to T.W. Computing hardware was supported by NIH Office of the Director grant (S10OD025181) via Brown's Center for Computation and Visualization (CCV). 
