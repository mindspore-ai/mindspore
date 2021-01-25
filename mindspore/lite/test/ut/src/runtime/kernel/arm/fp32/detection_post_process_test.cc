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

#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32/detection_post_process_fp32.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"
#include "src/common/file_utils.h"

namespace mindspore {
class TestDetectionPostProcessFp32 : public mindspore::CommonTest {
 public:
  TestDetectionPostProcessFp32() {}
};

void DetectionPostProcessTestInit(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                                  DetectionPostProcessParameter *param) {
  std::string input_boxes_path = "./test_data/detectionPostProcess/input_boxes.bin";
  size_t input_boxes_size;
  auto input_boxes_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(input_boxes_path.c_str(), &input_boxes_size));
  auto *input_boxes = new lite::Tensor;
  input_boxes->set_data_type(kNumberTypeFloat32);
  input_boxes->set_format(schema::Format_NHWC);
  input_boxes->set_shape({1, 1917, 4});
  input_boxes->MallocData();
  memcpy(input_boxes->MutableData(), input_boxes_data, input_boxes_size);
  inputs_->push_back(input_boxes);

  std::string input_scores_path = "./test_data/detectionPostProcess/input_scores.bin";
  size_t input_scores_size;
  auto input_scores_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(input_scores_path.c_str(), &input_scores_size));
  auto *input_scores = new lite::Tensor;
  input_scores->set_data_type(kNumberTypeFloat32);
  input_scores->set_format(schema::Format_NHWC);
  input_scores->set_shape({1, 1917, 91});
  input_scores->MallocData();
  memcpy(input_scores->MutableData(), input_scores_data, input_scores_size);
  inputs_->push_back(input_scores);

  std::string input_anchors_path = "./test_data/detectionPostProcess/input_anchors.bin";
  size_t input_anchors_size;
  auto input_anchors_data =
    reinterpret_cast<uint8_t *>(mindspore::lite::ReadFile(input_anchors_path.c_str(), &input_anchors_size));
  auto *input_anchors = new lite::Tensor;
  lite::QuantArg quant_arg;
  quant_arg.zeroPoint = 0;
  quant_arg.scale = 0.00645306;
  input_anchors->AddQuantParam(quant_arg);
  input_anchors->set_data_type(kNumberTypeUInt8);
  input_anchors->set_format(schema::Format_NHWC);
  input_anchors->set_shape({1917, 4});
  input_anchors->MallocData();
  memcpy(input_anchors->MutableData(), input_anchors_data, input_anchors_size);
  inputs_->push_back(input_anchors);

  auto *output_boxes = new lite::Tensor;
  output_boxes->set_data_type(kNumberTypeFloat32);
  output_boxes->set_shape({1, 10, 4});
  output_boxes->set_format(schema::Format_NHWC);
  output_boxes->MallocData();
  memset(output_boxes->MutableData(), 0, output_boxes->ElementsNum() * sizeof(float));

  auto *output_classes = new lite::Tensor;
  output_classes->set_data_type(kNumberTypeFloat32);
  output_classes->set_shape({1, 10});
  output_classes->set_format(schema::Format_NHWC);
  output_classes->MallocData();
  memset(output_classes->MutableData(), 0, output_classes->ElementsNum() * sizeof(float));

  auto *output_scores = new lite::Tensor;
  output_scores->set_data_type(kNumberTypeFloat32);
  output_scores->set_shape({1, 10});
  output_scores->set_format(schema::Format_NHWC);
  output_scores->MallocData();
  memset(output_scores->MutableData(), 0, output_scores->ElementsNum() * sizeof(float));

  auto *output_num_det = new lite::Tensor;
  output_num_det->set_data_type(kNumberTypeFloat32);
  output_num_det->set_shape({1});
  output_num_det->set_format(schema::Format_NHWC);
  output_num_det->MallocData();
  memset(output_num_det->MutableData(), 0, output_num_det->ElementsNum() * sizeof(float));

  outputs_->push_back(output_boxes);
  outputs_->push_back(output_classes);
  outputs_->push_back(output_scores);
  outputs_->push_back(output_num_det);

  param->h_scale_ = 5;
  param->w_scale_ = 5;
  param->x_scale_ = 10;
  param->y_scale_ = 10;
  param->nms_iou_threshold_ = 0.6;
  param->nms_score_threshold_ = 1e-8;
  param->max_detections_ = 10;
  param->detections_per_class_ = 100;
  param->max_classes_per_detection_ = 1;
  param->num_classes_ = 90;
  param->use_regular_nms_ = false;
  param->out_quantized_ = true;
}

TEST_F(TestDetectionPostProcessFp32, Fast) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto param = new DetectionPostProcessParameter();
  DetectionPostProcessTestInit(&inputs_, &outputs_, param);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::DetectionPostProcessCPUKernel *op =
    new kernel::DetectionPostProcessCPUKernel(reinterpret_cast<OpParameter *>(param), inputs_, outputs_, ctx);
  op->Init();
  op->Run();

  auto *output_boxes = reinterpret_cast<float *>(outputs_[0]->MutableData());
  size_t output_boxes_size;
  std::string output_boxes_path = "./test_data/detectionPostProcess/output_0.bin";
  auto correct_boxes =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(output_boxes_path.c_str(), &output_boxes_size));
  ASSERT_EQ(0, CompareOutputData(output_boxes, correct_boxes, outputs_[0]->ElementsNum(), 0.0001));

  auto *output_classes = reinterpret_cast<float *>(outputs_[1]->MutableData());
  size_t output_classes_size;
  std::string output_classes_path = "./test_data/detectionPostProcess/output_1.bin";
  auto correct_classes =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(output_classes_path.c_str(), &output_classes_size));
  ASSERT_EQ(0, CompareOutputData(output_classes, correct_classes, outputs_[1]->ElementsNum(), 0.0001));

  auto *output_scores = reinterpret_cast<float *>(outputs_[2]->MutableData());
  size_t output_scores_size;
  std::string output_scores_path = "./test_data/detectionPostProcess/output_2.bin";
  auto correct_scores =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(output_scores_path.c_str(), &output_scores_size));
  ASSERT_EQ(0, CompareOutputData(output_scores, correct_scores, outputs_[2]->ElementsNum(), 0.0001));

  auto *output_num_det = reinterpret_cast<float *>(outputs_[3]->MutableData());
  size_t output_num_det_size;
  std::string output_num_det_path = "./test_data/detectionPostProcess/output_3.bin";
  auto correct_num_det =
    reinterpret_cast<float *>(mindspore::lite::ReadFile(output_num_det_path.c_str(), &output_num_det_size));
  ASSERT_EQ(0, CompareOutputData(output_num_det, correct_num_det, outputs_[3]->ElementsNum(), 0.0001));

  delete op;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}
}  // namespace mindspore
