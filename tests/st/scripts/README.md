# how to run mindspore st

## Modify env config file

modify file `config/env_config.yaml`, the meaning of each field is as follows:

```yaml
pytest_path: ${HOME}/miniconda3/envs/py3.7/bin//pytest
python_path: ${HOME}/miniconda3/envs/py3.7/bin//python3
extend_envs: export PATH=${HOME}/miniconda3/envs/py3.9/bin:${PATH}; export PYTHONPATH=${HOME}/workspace/mindspore:${PYTHONPATH}
run_path: ${HOME}/workspace/mindspore/run
log_path: ${HOME}/workspace/mindspore/log
virtualenv:
  device_env_var_name: DEVICE_ID
  device_ids:
    - 0
  overall_networks: true
```

| field | value | description |
| ----- | ----- | ----------- |
| pytest_path | \${HOME}/miniconda3/envs/py3.7/bin//pytest | Path of pytest, if starts with XXX, the st test script will find it according to user's environment |
| python_path | \${HOME}/miniconda3/envs/py3.7/bin//pytest | Path of python, if starts with XXX, the st test script will find it according to user's environment |
| extend_envs | export PATH=\${HOME}/miniconda3/envs/py3.7/bin:\${PATH}; export PYTHONPATH=\${HOME}/workspace/mindspore:\${PYTHONPATH} | Commands to be executed before running each testcase, if starts with XXX, the st test script will set it according to user's configurations |
| run_path | /tmp/ms_run | Working path of running 1P testcase |
| log_path | /tmp/ms_log | Log path for saving logs of failed testcases |
| device_ids | a yaml list of type int | On which devices the mindspore st will run |
| overall_networks | true/false | When set to true and device_ids is 0~7 indicating this machine can run 8P testcases, otherwise only 1P will be exectuted |

## Requirements & Dependencies

- requirements

  ```bash
  pytest <= 6.0.0
  ```

- Dependencies

   ```bash
   pip install pycocotools matplotlib scikit-learn opencv-python easydict torch==1.12
   ```

## Run st testcases

- ascend910

  ```bash
  bash run_st.sh -p ascend -c ascend910 -l level0 -r ${HOME}/workspace/mindspore
  ```

- gpu

  ```bash
  # cuda-10.1
  bash run_st.sh -p gpu -c cuda-10.1 -l level0 -r ${HOME}/workspace/mindspore
  # cuda-11.1 or cuda-11.6
  bash run_st.sh -p gpu -c cuda-11.1 -l level0 -r ${HOME}/workspace/mindspore
  ```

## Running result

1. Commands of executing testcases are save to file `run_cmds.csv` in the same directory as `run_st.sh`.
2. Logs of failed testcase are saved to `log_path` configured in file `config/env_config.yaml`.
