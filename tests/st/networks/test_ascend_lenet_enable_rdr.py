import os
import tempfile
import json
import pytest

import mindspore.context as context
from .test_network_main import test_lenet

# create config file for RDR
def create_config_file(path):
    data_dict = {'rdr': {'enable': True, 'path': path}}
    filename = os.path.join(path, "mindspore_config.json")
    with open(filename, "w") as f:
        json.dump(data_dict, f)
    return filename

def test_train(device_type):
    context.set_context(mode=context.GRAPH_MODE, device_target=device_type)
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = create_config_file(tmpdir)
        context.set_context(env_config_path=config_file)
        test_lenet()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_train_with_Ascend():
    test_train("Ascend")
