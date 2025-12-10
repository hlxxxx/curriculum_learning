# curriculum_learning
- step1：
  'pip install torch transformers datasets accelerate peft modelscope'
- step2：accelerate config
-   This machine
-   multi-GPU
-   How many different machines will you use? 1
-   yes/NO 全选NO
-   How many GPU(s) should be used for distributed training? 2
-   What GPU(s)(by id) should be used for traininh on this machine as a comma-separated list? all
-   Do you wish to use mixed precision? bf16
- step3：
  'accelerate launch train.py'
- 可更改参数在config.py和train.py
