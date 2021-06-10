/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <memory>
#include "minddata/dataset/include/dataset/datasets.h"

using Dataset = mindspore::dataset::Dataset;
using Iterator = mindspore::dataset::Iterator;
using mindspore::dataset::Cifar10;
using mindspore::dataset::RandomSampler;
using mindspore::dataset::Tensor;

int main() {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10Dataset.";

  // Create a Cifar10 Dataset
  const int64_t num_samples = 10;
  std::string folder_path = "./testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, std::string(), RandomSampler(false, num_samples));

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  // Manually terminate the pipeline
  iter->Stop();
}
