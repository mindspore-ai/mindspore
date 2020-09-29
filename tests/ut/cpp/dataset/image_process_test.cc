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

#include "common/common.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "utils/log_adapter.h"

#include <fstream>

using namespace mindspore::dataset;
class MindDataImageProcess : public UT::Common {
 public:
  MindDataImageProcess() {}

  void SetUp() {}
};

void CompareMat(cv::Mat cv_mat, LiteMat lite_mat) {
  int cv_h = cv_mat.rows;
  int cv_w = cv_mat.cols;
  int cv_c = cv_mat.channels();
  int lite_h = lite_mat.height_;
  int lite_w = lite_mat.width_;
  int lite_c = lite_mat.channel_;
  ASSERT_TRUE(cv_h == lite_h);
  ASSERT_TRUE(cv_w == lite_w);
  ASSERT_TRUE(cv_c == lite_c);
}

LiteMat Lite3CImageProcess(LiteMat &lite_mat_bgr) {
  bool ret;
  LiteMat lite_mat_resize;
  ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
  if (!ret) {
    MS_LOG(ERROR) << "ResizeBilinear error";
  }
  LiteMat lite_mat_convert_float;
  ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0);
  if (!ret) {
    MS_LOG(ERROR) << "ConvertTo error";
  }

  LiteMat lite_mat_crop;
  ret = Crop(lite_mat_convert_float, lite_mat_crop, 16, 16, 224, 224);
  if (!ret) {
    MS_LOG(ERROR) << "Crop error";
  }

  std::vector<float> means = {0.485, 0.456, 0.406};
  std::vector<float> stds = {0.229, 0.224, 0.225};

  LiteMat lite_norm_mat_cut;
  SubStractMeanNormalize(lite_mat_crop, lite_norm_mat_cut, means, stds);

  return lite_norm_mat_cut;
}

cv::Mat cv3CImageProcess(cv::Mat &image) {
  cv::Mat resize_256_image;
  cv::resize(image, resize_256_image, cv::Size(256, 256), CV_INTER_LINEAR);
  cv::Mat float_256_image;
  resize_256_image.convertTo(float_256_image, CV_32FC3);

  cv::Mat roi_224_image;
  cv::Rect roi;
  roi.x = 16;
  roi.y = 16;
  roi.width = 224;
  roi.height = 224;

  float_256_image(roi).copyTo(roi_224_image);

  float meanR = 0.485;
  float meanG = 0.456;
  float meanB = 0.406;
  float varR = 0.229;
  float varG = 0.224;
  float varB = 0.225;
  cv::Scalar mean = cv::Scalar(meanR, meanG, meanB);
  cv::Scalar var = cv::Scalar(varR, varG, varB);

  cv::Mat imgMean(roi_224_image.size(), CV_32FC3, mean);
  cv::Mat imgVar(roi_224_image.size(), CV_32FC3, var);

  cv::Mat imgR1 = roi_224_image - imgMean;
  cv::Mat imgR2 = imgR1 / imgVar;
  return imgR2;
}

TEST_F(MindDataImageProcess, test3C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat cv_image = cv3CImageProcess(image);
  // cv::imwrite("/home/xlei/test_3cv.jpg", cv_image);

  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  if (!ret) {
    MS_LOG(ERROR) << "Init From RGBA error";
  }
  LiteMat lite_norm_mat_cut = Lite3CImageProcess(lite_mat_bgr);

  cv::Mat dst_image(lite_norm_mat_cut.height_, lite_norm_mat_cut.width_, CV_32FC3, lite_norm_mat_cut.data_ptr_);
  //  cv::imwrite("/home/xlei/test_3clite.jpg", dst_image);

  CompareMat(cv_image, lite_norm_mat_cut);
}

LiteMat Lite1CImageProcess(LiteMat &lite_mat_bgr) {
  LiteMat lite_mat_resize;
  ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
  LiteMat lite_mat_convert_float;
  ConvertTo(lite_mat_resize, lite_mat_convert_float);

  LiteMat lite_mat_cut;

  Crop(lite_mat_convert_float, lite_mat_cut, 16, 16, 224, 224);

  std::vector<float> means = {0.485};
  std::vector<float> stds = {0.229};

  LiteMat lite_norm_mat_cut;

  SubStractMeanNormalize(lite_mat_cut, lite_norm_mat_cut, means, stds);
  return lite_norm_mat_cut;
}

cv::Mat cv1CImageProcess(cv::Mat &image) {
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, CV_BGR2GRAY);

  cv::Mat resize_256_image;
  cv::resize(gray_image, resize_256_image, cv::Size(256, 256), CV_INTER_LINEAR);
  cv::Mat float_256_image;
  resize_256_image.convertTo(float_256_image, CV_32FC3);

  cv::Mat roi_224_image;
  cv::Rect roi;
  roi.x = 16;
  roi.y = 16;
  roi.width = 224;
  roi.height = 224;

  float_256_image(roi).copyTo(roi_224_image);

  float meanR = 0.485;
  float varR = 0.229;
  cv::Scalar mean = cv::Scalar(meanR);
  cv::Scalar var = cv::Scalar(varR);

  cv::Mat imgMean(roi_224_image.size(), CV_32FC1, mean);
  cv::Mat imgVar(roi_224_image.size(), CV_32FC1, var);

  cv::Mat imgR1 = roi_224_image - imgMean;
  cv::Mat imgR2 = imgR1 / imgVar;
  return imgR2;
}

TEST_F(MindDataImageProcess, test1C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat cv_image = cv1CImageProcess(image);

  // cv::imwrite("/home/xlei/test_c1v.jpg", cv_image);

  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  LiteMat lite_norm_mat_cut = Lite1CImageProcess(lite_mat_bgr);
  cv::Mat dst_image(lite_norm_mat_cut.height_, lite_norm_mat_cut.width_, CV_32FC1, lite_norm_mat_cut.data_ptr_);
  // cv::imwrite("/home/xlei/test_c1lite.jpg", dst_image);

  CompareMat(cv_image, lite_norm_mat_cut);
}

TEST_F(MindDataImageProcess, TestPadd) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat resize_256_image;
  cv::resize(image, resize_256_image, cv::Size(256, 256), CV_INTER_LINEAR);
  int left = 10;
  int right = 10;
  int top = 10;
  int bottom = 10;
  cv::Mat b_image;
  cv::Scalar color = cv::Scalar(255, 255, 255);
  cv::copyMakeBorder(resize_256_image, b_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  // cv::imwrite("/home/xlei/test_ccc.jpg", b_image);
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

  LiteMat lite_mat_resize;
  ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);

  LiteMat makeborder;
  Pad(lite_mat_resize, makeborder, top, bottom, left, right, PaddBorderType::PADD_BORDER_CONSTANT, 255, 255, 255);

  cv::Mat dst_image(256 + top + bottom, 256 + left + right, CV_8UC3, makeborder.data_ptr_);

  // cv::imwrite("/home/xlei/test_liteccc.jpg", dst_image);
}

TEST_F(MindDataImageProcess, TestGetDefaultBoxes) {
  std::string benchmark = "data/dataset/testLite/default_boxes.bin";
  BoxesConfig config;
  config.img_shape = {300, 300};
  config.num_default = {3, 6, 6, 6, 6, 6};
  config.feature_size = {19, 10, 5, 3, 2, 1};
  config.min_scale = 0.2;
  config.max_scale = 0.95;
  config.aspect_rations = {{2}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}};
  config.steps = {16, 32, 64, 100, 150, 300};
  config.prior_scaling = {0.1, 0.2};

  int rows = 1917;
  int cols = 4;
  std::vector<double> benchmark_boxes(rows * cols);
  std::ifstream in(benchmark, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(benchmark_boxes.data()), benchmark_boxes.size() * sizeof(double));
  in.close();

  std::vector<std::vector<float>> default_boxes = GetDefaultBoxes(config);
  EXPECT_EQ(default_boxes.size(), rows);
  EXPECT_EQ(default_boxes[0].size(), cols);

  double distance = 0.0f;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      distance += pow(default_boxes[i][j] - benchmark_boxes[i * cols + j], 2);
    }
  }
  distance = sqrt(distance);
  EXPECT_LT(distance, 1e-5);
}

TEST_F(MindDataImageProcess, TestApplyNms) {
  std::vector<std::vector<float>> all_boxes = {{1, 1, 2, 2}, {3, 3, 4, 4}, {5, 5, 6, 6}, {5, 5, 6, 6}};
  std::vector<float> all_scores = {0.6, 0.5, 0.4, 0.9};
  std::vector<int> keep = ApplyNms(all_boxes, all_scores, 0.5, 10);
  ASSERT_TRUE(keep[0] == 3);
  ASSERT_TRUE(keep[1] == 0);
  ASSERT_TRUE(keep[2] == 1);
}

TEST_F(MindDataImageProcess, TestAffineInput) {
  LiteMat src(3, 3);
  LiteMat dst;
  double M[6] = {1};
  EXPECT_FALSE(Affine(src, dst, M, {}, UINT8_C1(0)));
  EXPECT_FALSE(Affine(src, dst, M, {3}, UINT8_C1(0)));
  EXPECT_FALSE(Affine(src, dst, M, {0, 0}, UINT8_C1(0)));
}

TEST_F(MindDataImageProcess, TestAffine) {
  // The input matrix
  // 0 0 1 0 0
  // 0 0 1 0 0
  // 2 2 3 2 2
  // 0 0 1 0 0
  // 0 0 1 0 0
  size_t rows = 5;
  size_t cols = 5;
  LiteMat src(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (i == 2 && j == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 3;
      } else if (i == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 2;
      } else if (j == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 1;
      } else {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 0;
      }
    }
  }

  // Expect output matrix
  // 0 0 2 0 0
  // 0 0 2 0 0
  // 1 1 3 1 1
  // 0 0 2 0 0
  // 0 0 2 0 0
  LiteMat expect(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (i == 2 && j == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 3;
      } else if (i == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 1;
      } else if (j == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 2;
      } else {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 0;
      }
    }
  }

  double angle = 90.0f;
  cv::Point2f center(rows / 2, cols / 2);
  cv::Mat rotate_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
  double M[6];
  for (size_t i = 0; i < 6; i++) {
    M[i] = rotate_matrix.at<double>(i);
  }
  std::cout << std::endl;
  LiteMat dst;
  EXPECT_TRUE(Affine(src, dst, M, {rows, cols}, UINT8_C1(0)));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      EXPECT_EQ(static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j].c1,
                static_cast<UINT8_C1 *>(dst.data_ptr_)[i * cols + j].c1);
    }
  }
}
