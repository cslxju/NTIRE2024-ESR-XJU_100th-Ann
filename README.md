# [NTIRE 2024 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2024/) @ [CVPR 2024](https://cvpr.thecvf.com/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2024_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## How to test the baseline model?

1. `git clone https://github.com/daydreamer2024/NTIRE2024-ESR-XJU_100th-Ann.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 31
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.

As a reference, we provide the results of RLFN (baseline method) below:
- Average PSNR on DIV2K_LSDIR_valid: 26.96 dB
- Average PSNR on DIV2K_LSDIR_test: 27.07 dB
- Number of parameters: 0.317 M
- Runtime: 13.54 ms (Average runtime of 16.18 ms on DIV2K_LSDIR_valid data and 10.89 ms on DIV2K_LSDIR_test data)
- FLOPs on an LR image of size 256×256: 19.67 G

    Please note that the results reported above are the average of 5 runs, and each run is conducted on the same device (i.e., NVIDIA GeForce RTX 3090 GPU).


## How to add your model to this baseline?

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1ZFlte0uR4bNl6UVJxShESkui1n3ejzXAvUX_e1qyhSc/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RLFN import RLFN_Prune
    from fvcore.nn import FlopCountAnalysis

    model = RLFN_Prune()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # The FLOPs calculation in previous NTIRE_ESR Challenge
    # flops = get_model_flops(model, input_dim, False)
    # flops = flops / 10 ** 9
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # fvcore is used in NTIRE2024_ESR for FLOPs calculation
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops/10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## How the Ranking Strategy Works?

After the organizers receive all the submitted codes/checkpoints/results, four steps are adopted for the evaluation:

- Step1: The organizers will execute each model five times to reevaluate all submitted methods on the same device, specifically the NVIDIA GeForce RTX 3090. The average results of these five runs will be documented for each metric.
- Step2: To ensure PSNR consistency with the baseline method RLFN, PSNR checks will be conducted for all submitted methods. Any method with a PSNR below 26.90 dB on the DIV2K_LSDIR_valid dataset or less than 26.99 on the DIV2K_LSDIR_test datasets will be excluded from the comparison list for the remaining rankings. 
- Step3: For the rest, the *Score_Runtime*, *Score_FLOPs*, and the *Score_Params* will be calculated as follows:

```
     Score_Runtime = exp(2*Runtime / Runtime_RLFN)
    
     Score_FLOPs = exp(2*FLOPs / FLOPs_RLFN)
     
     Score_Params = exp(2*Params / Params_RLFN)
```
-   Step4: The final comparison score will be calculated as follows:
```
    Score_Final = 0.7*Score_Runtime + 0.15*Score_FLOPs + 0.15*Score_Params
```
Let's take the baseline as an example, given the results (i.e., average Runtime_RLFN = 13.54 ms, FLOPs_RLFN = 19.67 G, and Params_RLFN = 0.317 M) of RLFN, we have:
```
    Score_Runtime = 7.3891
    Score_FLOPs = 7.3891
    Score_Params = 7.3891
    Score_Final = 7.3891 
```
:heavy_exclamation_mark:The ranking for each sub-track will be generated based on the corresponding Score (i.e., *Score_Runtime*, *Score_FLOPs*, and *Score_Params*), while for the main track, the ranking will be determined by the *Score_Final*.

If you have any question, feel free to reach out the contact persons and direct managers of the NTIRE challenge.


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
