CosPar-Scheduler  

CosineAnnealingWarmPartialRestarts：コスパスケジューラ  

PyTorch向けのカスタム学習率スケジューラーです、  
｢途中のサイクルは学習率を指定の最低値(min_lr_rate)を維持し、  
最後のサイクルだけは通常通り0まで落とす｣という、  
部分的なリスタート(Partial Restarts)を行います、

readme：[English](README.md) | [日本語](README_JA.md)  

<img width="800" height="400" alt="cospar" src="https://github.com/user-attachments/assets/1c609f34-9815-4347-98ad-e31773f49113" />

主な特徴  

  部分的なリスタート(Partial Restarts)：  
      中間サイクルは、学習率が min_lr_rate(例： 0.3、0.5、など) を下回らないように、  
      コサインカーブで減衰させます、これにより、ほぼ０の無学習を抑制しつつリスタートできます、  

  最後サイクルは通常の減衰：  
      最終サイクルのみ、通常のコサインアニーリングと同様に学習率を 0.0 まで減衰します、  

  ウォームアップ対応：  
      num_warmup_steps を指定することで、初期のウォームアップ期間もサポートします、  

  柔軟な統合：  
      PyTorchの _LRScheduler を継承しているため、標準的なPyTorchのコードはもちろん、  
      引数指定式のスクリプトでもカスタムスケジューラーとして簡単に組み込めます、  

License  
Licensed under the Apache License 2.0. Feel free to use, modify, and distribute.  