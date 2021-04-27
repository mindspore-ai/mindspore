/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.dance.algorithm;

import android.util.Log;

import com.huawei.hms.mlsdk.skeleton.MLSkeleton;

import java.util.ArrayList;
import java.util.List;

import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_LEFT_ANKLE;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_LEFT_ELBOW;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_LEFT_KNEE;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_LEFT_WRIST;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_NECK;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_RIGHT_ELBOW;
import static com.huawei.hms.mlsdk.skeleton.MLJoint.TYPE_RIGHT_WRIST;

public class ModelDataUtils {
    public static final int NO_POINT = -2;
    public static final int NO_ACT = -1;

    private static final String TAG = ModelDataUtils.class.getSimpleName();
    private static final double[] MODEL_DATA_ARRAY = new double[]{
            1, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            2, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            3, -0.447213595, -0.447213595, 0.5, -0.447213595, -0.948683298, -0.707106781,
            4, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            5, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            6, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            7, -0.447213595, -0.447213595, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            8, -0.447213595, -0.447213595, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            9, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            10, -0.447213595, -0.707106781, 0.5, 0.5, -0.948683298, -0.707106781,
            11, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            12, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            13, -0.707106781, -0.707106781, 0.5, -1, -1, -0.948683298,
            14, -0.707106781, -0.707106781, 0.5, -0.707106781, -1, -0.948683298,
            15, -0.707106781, -0.447213595, 0.5, -1, -1, -0.948683298,
            16, -0.707106781, -0.447213595, 0.5, -0.707106781, -1, -0.948683298,
            17, -0.707106781, -0.447213595, 0.5, -0.707106781, -1, -0.948683298,
            18, -0.447213595, -0.447213595, -0.707106781, -0.707106781, -0.948683298, -0.707106781,
            19, -0.707106781, -0.707106781, -0.894427191, -1, -0.948683298, -0.832050294,
            20, -0.447213595, -0.707106781, -0.707106781, -0.447213595, -0.948683298, -0.948683298,
            21, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            22, -0.316227766, -0.316227766, -0.371390676, -0.447213595, -0.948683298, -0.6,
            23, -0.447213595, -0.447213595, -0.196116135, -0.447213595, -0.948683298, -0.948683298,
            24, -0.447213595, -0.707106781, -0.316227766, -1, -1, -0.832050294,
            25, -0.447213595, -0.707106781, -0.316227766, -1, -1, -0.832050294,
            26, -1, -1, -1, 0.5, -0.948683298, -0.948683298,
            27, -1, -1, -1, -0.894427191, -0.948683298, -0.948683298,
            28, -0.447213595, -0.447213595, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            29, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            30, -0.447213595, 0.624695048, -0.196116135, 0.9701425, -0.948683298, -0.707106781,
            31, -0.447213595, 0.514495755, -0.196116135, 0.948683298, -0.948683298, -0.707106781,
            32, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            33, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            34, -0.447213595, -0.707106781, -0.196116135, -0.316227766, -0.948683298, -0.707106781,
            35, -0.447213595, 0.554700196, -0.196116135, 0.948683298, -0.948683298, -0.707106781,
            36, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.948683298,
            37, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            38, -0.447213595, -0.316227766, -0.196116135, -0.316227766, -0.948683298, -0.707106781,
            39, -0.447213595, -0.554700196, -0.196116135, -0.316227766, -0.948683298, -0.707106781,
            40, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            41, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            42, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            43, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            44, -0.447213595, -0.447213595, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            45, -0.447213595, -0.707106781, -0.196116135, -0.554700196, -0.948683298, -0.707106781,
            46, -0.447213595, -0.447213595, -0.196116135, -0.894427191, -0.948683298, -0.707106781,
            47, -0.447213595, -0.554700196, -0.196116135, -0.894427191, -0.948683298, -0.707106781,
            48, -0.707106781, -0.447213595, -0.447213595, -0.707106781, -0.948683298, -0.894427191,
            49, -0.447213595, -0.447213595, -0.447213595, -0.707106781, -0.948683298, -0.948683298,
            50, -0.447213595, -0.447213595, -0.447213595, -0.707106781, -0.948683298, -0.948683298,
            51, -0.447213595, -0.707106781, -0.447213595, -0.707106781, -0.948683298, -0.948683298,
            52, -0.447213595, -0.447213595, -0.447213595, -0.894427191, -0.948683298, -1,
            53, -0.447213595, -0.447213595, 0.707106781, -0.447213595, -0.948683298, -0.707106781,
            54, -0.707106781, -0.707106781, 0.5, -0.707106781, -0.948683298, -0.832050294,
            55, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            56, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            57, -0.316227766, -0.447213595, -0.316227766, -0.316227766, -0.948683298, -0.6,
            58, -0.316227766, -0.447213595, -0.316227766, -0.316227766, -0.948683298, -0.707106781,
            59, -0.447213595, -0.707106781, -0.242535625, -0.707106781, -0.948683298, -0.707106781,
            60, -0.447213595, -0.707106781, -0.242535625, -0.707106781, -0.948683298, -0.707106781,
            61, -0.447213595, -0.707106781, -0.242535625, -0.707106781, -0.948683298, -0.832050294,
            62, -0.447213595, -0.707106781, -0.242535625, -0.894427191, -0.948683298, -0.832050294,
            63, -0.447213595, -0.707106781, -0.242535625, -0.894427191, -0.948683298, -0.707106781,
            64, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            65, -0.447213595, -0.447213595, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            66, -0.316227766, -0.316227766, -0.371390676, -0.447213595, -0.948683298, -0.6,
            67, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            68, -0.447213595, -0.707106781, -0.196116135, -0.707106781, -0.948683298, -0.707106781,
            69, -0.447213595, -0.707106781, -0.196116135, -0.447213595, -0.948683298, -0.707106781,
            70, -0.447213595, -0.447213595, -0.196116135, -0.707106781, -0.948683298, -0.707106781};

    private static final float NO_POINT_SCALE = 0.4f;
    private static final float NO_ACT_SCALE = 0.4f;

    public static List<ModelDataBean> getModelDataList() {
        List<ModelDataBean> modelDataBeanList = new ArrayList<>();
        for (int i = 0; i < MODEL_DATA_ARRAY.length; i += 7) {
            if (i % 7 == 0) {
                ModelDataBean bean = new ModelDataBean();
                bean.setId((int) MODEL_DATA_ARRAY[i]);
                bean.setSinRightElbow(MODEL_DATA_ARRAY[i + 1]);
                bean.setSinRightWrist(MODEL_DATA_ARRAY[i + 2]);
                bean.setSinLeftElbow(MODEL_DATA_ARRAY[i + 3]);
                bean.setSinLeftWrist(MODEL_DATA_ARRAY[i + 4]);
                bean.setSinLeftKnee(MODEL_DATA_ARRAY[i + 5]);
                bean.setSinLeftAnkle(MODEL_DATA_ARRAY[i + 6]);
                modelDataBeanList.add(bean);
            }
        }
        return modelDataBeanList;
    }

    public static ModelDataBean hmsData2ModelData(MLSkeleton mlSkeleton) {

        ModelDataBean bean = new ModelDataBean();

        float rightElbowA = mlSkeleton.getJointPoint(TYPE_RIGHT_ELBOW).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float rightElbowB = mlSkeleton.getJointPoint(TYPE_RIGHT_ELBOW).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float rightElbowC = (float) Math.sqrt(Math.pow(rightElbowA, 2) + Math.pow(rightElbowB, 2));
        if (mlSkeleton.getJointPoint(TYPE_RIGHT_ELBOW).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_RIGHT_ELBOW).getPointX() == 0) {
            bean.setRightElbowXY(1);
        } else {
            bean.setRightElbowXY(0);
        }
        bean.setSinRightElbow(rightElbowC == 0 ? 0.5 : (rightElbowA / rightElbowC));

        float rightWristA = mlSkeleton.getJointPoint(TYPE_RIGHT_WRIST).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float rightWristB = mlSkeleton.getJointPoint(TYPE_RIGHT_WRIST).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float rightWristC = (float) Math.sqrt(Math.pow(rightWristA, 2) + Math.pow(rightWristB, 2));
        if (mlSkeleton.getJointPoint(TYPE_RIGHT_WRIST).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_RIGHT_WRIST).getPointX() == 0) {
            bean.setRightWristXY(1);
        } else {
            bean.setRightWristXY(0);
        }
        bean.setSinRightWrist(rightWristC == 0 ? 0.5 : (rightWristA / rightWristC));

        float leftElbowA = mlSkeleton.getJointPoint(TYPE_LEFT_ELBOW).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float leftElbowB = mlSkeleton.getJointPoint(TYPE_LEFT_ELBOW).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float leftElbowC = (float) Math.sqrt(Math.pow(leftElbowA, 2) + Math.pow(leftElbowB, 2));
        if (mlSkeleton.getJointPoint(TYPE_LEFT_ELBOW).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_LEFT_ELBOW).getPointX() == 0) {
            bean.setLeftElbowXY(1);
        } else {
            bean.setLeftElbowXY(0);
        }
        bean.setSinLeftElbow(leftElbowC == 0 ? 0.5 : (leftElbowA / leftElbowC));

        float leftWristA = mlSkeleton.getJointPoint(TYPE_LEFT_WRIST).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float leftWristB = mlSkeleton.getJointPoint(TYPE_LEFT_WRIST).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float leftWristC = (float) Math.sqrt(Math.pow(TYPE_LEFT_WRIST, 2) + Math.pow(leftWristB, 2));
        if (mlSkeleton.getJointPoint(TYPE_LEFT_WRIST).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_LEFT_WRIST).getPointX() == 0) {
            bean.setLeftWristXY(1);
        } else {
            bean.setLeftWristXY(0);
        }
        bean.setSinLeftWrist(leftWristC == 0 ? 0.5 : (leftWristA / leftWristC));

        float leftKneeA = mlSkeleton.getJointPoint(TYPE_LEFT_KNEE).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float leftKneeB = mlSkeleton.getJointPoint(TYPE_LEFT_KNEE).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float leftKneeC = (float) Math.sqrt(Math.pow(leftKneeA, 2) + Math.pow(leftKneeB, 2));
        if (mlSkeleton.getJointPoint(TYPE_LEFT_KNEE).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_LEFT_KNEE).getPointX() == 0) {
            bean.setLeftKneeXY(1);
        } else {
            bean.setLeftKneeXY(0);
        }
        bean.setSinLeftKnee(leftKneeC == 0 ? 0.5 : (leftKneeA / leftKneeC));

        float leftAnkleA = mlSkeleton.getJointPoint(TYPE_LEFT_ANKLE).getPointY()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointY();
        float leftAnkleB = mlSkeleton.getJointPoint(TYPE_LEFT_ANKLE).getPointX()
                - mlSkeleton.getJointPoint(TYPE_NECK).getPointX();
        float leftAnkleC = (float) Math.sqrt(Math.pow(leftAnkleA, 2) + Math.pow(leftAnkleB, 2));
        if (mlSkeleton.getJointPoint(TYPE_LEFT_ANKLE).getPointY() == 0 && mlSkeleton.getJointPoint(TYPE_LEFT_ANKLE).getPointX() == 0) {
            bean.setLeftAnkleXY(1);
        } else {
            bean.setLeftAnkleXY(0);
        }
        bean.setSinLeftAnkle(leftAnkleC == 0 ? 0.5 : (leftAnkleA / leftAnkleC));
        Log.d(TAG, "hmsData2ModelData"
                + ", getSinRightElbow=" + bean.getSinRightElbow()
                + ", getSinRightWrist=" + bean.getSinRightWrist()
                + ", getSinLeftElbow=" + bean.getSinLeftElbow()
                + ", getSinLeftWrist=" + bean.getSinLeftWrist()
                + ", getSinLeftKnee=" + bean.getSinLeftKnee()
                + ", getSinLeftAnkle=" + bean.getSinLeftAnkle()

        );
        return bean;
    }


    public static double getScore(List<ModelDataBean> realDataList) {
        List<ModelDataBean> originList = getModelDataList();
        if (realDataList.size() > originList.size()) {
            for (int i = originList.size(); i < realDataList.size(); i++) {
                realDataList.remove(i);
            }
        }

        double sum = 0;
        int noDataNumble = 0;

        int sumRightElbowXY = 0;
        int sumRightWristXY = 0;
        int sumLeftElbowXY = 0;
        int sumLeftWristXY = 0;
        int sumLeftKneeXY = 0;
        int sumLeftAnkleXY = 0;

        for (int i = 0; i < realDataList.size(); i++) {
            ModelDataBean realBean = realDataList.get(i);
            ModelDataBean originBean = originList.get(i);
            if (realBean == null) {
                noDataNumble++;
                Log.d(TAG, "getScore, i= " + i + ", this data is null");
                continue;
            }
            double rightElbow = Math.abs(originBean.getSinRightElbow()) == 0.5 ?
                    Math.abs(originBean.getSinRightElbow() - realBean.getSinRightElbow() - 0.5) / 2 :
                    Math.abs(originBean.getSinRightElbow() - realBean.getSinRightElbow()) / 2;
            double rightWrist = Math.abs(originBean.getSinRightWrist()) == 0.5 ?
                    Math.abs(originBean.getSinRightWrist() - realBean.getSinRightWrist() - 0.5) / 2 :
                    Math.abs(originBean.getSinRightWrist() - realBean.getSinRightWrist()) / 2;
            double leftElbow = Math.abs(originBean.getSinLeftElbow()) == 0.5 ?
                    Math.abs(originBean.getSinLeftElbow() - realBean.getSinLeftElbow() - 0.5) / 2 :
                    Math.abs(originBean.getSinLeftElbow() - realBean.getSinLeftElbow()) / 2;
            double leftWrist = Math.abs(originBean.getSinLeftWrist()) == 0.5 ?
                    Math.abs(originBean.getSinLeftWrist() - realBean.getSinLeftWrist() - 0.5) / 2 :
                    Math.abs(originBean.getSinLeftWrist() - realBean.getSinLeftWrist()) / 2;
            double leftKnee = Math.abs(originBean.getSinLeftKnee()) == 0.5 ?
                    Math.abs(originBean.getSinLeftKnee() - realBean.getSinLeftKnee() - 0.5) / 2 :
                    Math.abs(originBean.getSinLeftKnee() - realBean.getSinLeftKnee()) / 2;
            double leftAnkle = Math.abs(originBean.getSinLeftAnkle()) == 0.5 ?
                    Math.abs(originBean.getSinLeftAnkle() - realBean.getSinLeftAnkle() - 0.5) / 2 :
                    Math.abs(originBean.getSinLeftAnkle() - realBean.getSinLeftAnkle()) / 2;
            double total = rightElbow + rightWrist + leftElbow + leftWrist + leftKnee + leftAnkle;
            sum += total;

            sumRightElbowXY += realBean.getRightElbowXY();
            sumRightWristXY += realBean.getRightWristXY();
            sumLeftElbowXY += realBean.getLeftElbowXY();
            sumLeftWristXY += realBean.getLeftWristXY();
            sumLeftKneeXY += realBean.getLeftKneeXY();
            sumLeftAnkleXY += realBean.getLeftAnkleXY();

            Log.d(TAG, "getScore, i= " + i
                    + ", total=" + total
                    + ", rightElbow=" + rightElbow
                    + ", rightWrist=" + rightWrist
                    + ", leftElbow=" + leftElbow
                    + ", leftWrist=" + leftWrist
                    + ", leftKnee=" + leftKnee
                    + ", leftAnkle=" + leftAnkle);
        }


        float onPointValue = (realDataList.size() - noDataNumble) * NO_POINT_SCALE;
        if (sumRightElbowXY >= onPointValue || sumRightWristXY >= onPointValue || sumLeftElbowXY >= onPointValue ||
                sumLeftWristXY >= onPointValue || sumLeftKneeXY >= onPointValue || sumLeftAnkleXY > +onPointValue) {
            return NO_POINT;   // Invalid point position scanned
        } else {
            double StandardDeviation = StandardDeviation(realDataList);
            Log.e(TAG, "getScore, StandardDeviation= " + StandardDeviation);
            if (StandardDeviation <= NO_ACT_SCALE) {
                return NO_ACT;
            } else {
                double average = sum / (realDataList.size() - noDataNumble) / 6;
                Log.e(TAG, "getScore, average= " + average
                        + ", sum=" + sum
                        + ", noDataNumble=" + noDataNumble
                        + ", realDataList.size()=" + realDataList.size());

                double score = 100 - 120 * (average - 0.5);
                if (score >= 100) {
                    score = 98;
                } else if (score < 60) {
                    score = 63;
                }
                return score;
            }
        }
    }

    // standard deviation σ=sqrt(s^2)
    public static double StandardDeviation(List<ModelDataBean> realDataList) {
        int frameNum = realDataList.size();
        int nullSum = 0;
        double sum = 0;
        for (int i = 0; i < frameNum; i++) {// Sum
            if (realDataList.get(i) != null) {
                ModelDataBean bean = realDataList.get(i);
                if (bean.getSinRightElbow() == 0.5 || bean.getSinRightElbow() == 0.5 || bean.getSinLeftElbow() == 0.5 ||
                        bean.getSinLeftWrist() == 0.5 || bean.getSinLeftKnee() == 0.5 || bean.getSinLeftAnkle() == 0.5) {
                    nullSum++;
                } else {
                    sum += bean.getSinRightElbow() + bean.getSinRightElbow()
                            + bean.getSinLeftElbow() + bean.getSinLeftWrist()
                            + bean.getSinLeftKnee() + bean.getSinLeftAnkle();
                }
            } else {
                nullSum++;
            }
        }
        double dAve = sum / (frameNum - nullSum);// Averaging
        double dVar = 0;
        for (int i = 0; i < frameNum; i++) {// Seek variance
            if (realDataList.get(i) != null) {
                ModelDataBean bean = realDataList.get(i);
                if (bean.getSinRightElbow() != 0.5 && bean.getSinRightElbow() != 0.5 && bean.getSinLeftElbow() != 0.5 &&
                        bean.getSinLeftWrist() != 0.5 && bean.getSinLeftKnee() != 0.5 && bean.getSinLeftAnkle() != 0.5) {
                    double sumSin = bean.getSinRightElbow() + bean.getSinRightElbow()
                            + bean.getSinLeftElbow() + bean.getSinLeftWrist()
                            + bean.getSinLeftKnee() + bean.getSinLeftAnkle();
                    dVar += (sumSin - dAve) * (sumSin - dAve);
                }
            }
        }
        Log.e(TAG, "getScore, Standard111  nullSum= " + nullSum);
        return Math.sqrt(dVar / (frameNum - nullSum));
    }
}
