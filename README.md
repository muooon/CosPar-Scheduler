CosPar-Scheduler  

CosineAnnealingWarmPartialRestarts: CosPar Scheduler  

A custom learning rate scheduler for PyTorch.  
It performs "partial restarts" where "intermediate cycles maintain the learning rate at a specified minimum value (min_lr_rate), while only the final cycle decays normally down to 0."  

readme：[English](README.md) | [日本語](README_JA.md)  

<img width="800" height="400" alt="cospar" src="https://github.com/user-attachments/assets/1c609f34-9815-4347-98ad-e31773f49113" />

Key Features  

  Partial Restarts:  
  Intermediate cycles decay via a cosine curve such that the learning rate does not fall below min_lr_rate (e.g., 0.3, 0.5). This allows for restarts while mitigating near-zero stagnant training phases.  

  Normal Decay in Final Cycle:  
  Only the final cycle decays the learning rate all the way down to 0.0, just like standard cosine annealing.  

  Warmup Support:  
  Supports an initial warmup period by specifying num_warmup_steps.  

  Flexible Integration:  
  Inheriting from PyTorch's _LRScheduler, it can be easily integrated as a custom scheduler into standard PyTorch code as well as argument-driven training scripts.  

License  
Licensed under the Apache License 2.0. Feel free to use, modify, and distribute.  