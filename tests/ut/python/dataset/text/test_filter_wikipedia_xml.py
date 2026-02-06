# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Testing FilterWikipediaXML op
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger


def count_unequal_element(data_expected, data_me):
    assert data_expected.shape == data_me.shape
    assert data_expected == data_me


def test_filter_wikipedia_xml_eager():
    """
    Feature: FilterWikipediaXML
    Description: Test FilterWikipediaXML in eager mode
    Expectation: The data is processed successfully
    """
    logger.info("test FilterWikipediaXML in eager mode")

    # Original text
    input01 = np.array(["Welcome to China"], dtype=np.unicode_)
    # Expect text
    expected = np.array(["welcome to china"], dtype=np.unicode_)
    filter_wikipedia_xml_op = text.FilterWikipediaXML()
    output = filter_wikipedia_xml_op(input01)
    count_unequal_element(expected, output)


def test_filter_wikipedia_xml_pipeline():
    """
    Feature: FilterWikipediaXML
    Description: Test FilterWikipediaXML in pipeline mode
    Expectation: The data is processed successfully
    """
    logger.info("test FilterWikipediaXML in pipeline mode")

    # Original text
    input02 = np.array(["Welcome to China", "中国", "ABC"])
    # Expect text
    expected = np.array(["welcome to china", "", "abc"])
    dataset = ds.NumpySlicesDataset(input02, ["text"], shuffle=False)
    filter_wikipedia_xml_op = text.FilterWikipediaXML()
    # Filtered waveform by filter_wikipedia_xml
    dataset = dataset.map(input_columns=["text"], operations=filter_wikipedia_xml_op, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(np.array(expected[i]), data['text'])
        i += 1


if __name__ == "__main__":
    test_filter_wikipedia_xml_eager()
    test_filter_wikipedia_xml_pipeline()
