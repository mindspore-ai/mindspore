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

public class ModelDataBean {
    // RightElbow cos, RightWrist cos, LeftElbow cos, LeftWrist cos, LeftKnee cos, LeftAnkle cos
    private int id;
    private double sinRightElbow;
    private double sinRightWrist;
    private double sinLeftElbow;
    private double sinLeftWrist;
    private double sinLeftKnee;
    private double sinLeftAnkle;

    private int rightElbowXY;
    private int rightWristXY;
    private int leftElbowXY;
    private int leftWristXY;
    private int leftKneeXY;
    private int leftAnkleXY;

    public int getRightElbowXY() {
        return rightElbowXY;
    }

    public ModelDataBean setRightElbowXY(int rightElbowXY) {
        this.rightElbowXY = rightElbowXY;
        return this;
    }

    public int getRightWristXY() {
        return rightWristXY;
    }

    public ModelDataBean setRightWristXY(int rightWristXY) {
        this.rightWristXY = rightWristXY;
        return this;
    }

    public int getLeftElbowXY() {
        return leftElbowXY;
    }

    public ModelDataBean setLeftElbowXY(int leftElbowXY) {
        this.leftElbowXY = leftElbowXY;
        return this;
    }

    public int getLeftWristXY() {
        return leftWristXY;
    }

    public ModelDataBean setLeftWristXY(int leftWristXY) {
        this.leftWristXY = leftWristXY;
        return this;
    }

    public int getLeftKneeXY() {
        return leftKneeXY;
    }

    public ModelDataBean setLeftKneeXY(int leftKneeXY) {
        this.leftKneeXY = leftKneeXY;
        return this;
    }

    public int getLeftAnkleXY() {
        return leftAnkleXY;
    }

    public ModelDataBean setLeftAnkleXY(int leftAnkleXY) {
        this.leftAnkleXY = leftAnkleXY;
        return this;
    }

    public ModelDataBean() {
    }

    public double getSinRightElbow() {
        return sinRightElbow;
    }

    public ModelDataBean setSinRightElbow(double sinRightElbow) {
        this.sinRightElbow = sinRightElbow;
        return this;
    }

    public double getSinRightWrist() {
        return sinRightWrist;
    }

    public ModelDataBean setSinRightWrist(double sinRightWrist) {
        this.sinRightWrist = sinRightWrist;
        return this;
    }

    public double getSinLeftElbow() {
        return sinLeftElbow;
    }

    public ModelDataBean setSinLeftElbow(double sinLeftElbow) {
        this.sinLeftElbow = sinLeftElbow;
        return this;
    }

    public double getSinLeftWrist() {
        return sinLeftWrist;
    }

    public ModelDataBean setSinLeftWrist(double sinLeftWrist) {
        this.sinLeftWrist = sinLeftWrist;
        return this;
    }

    public double getSinLeftKnee() {
        return sinLeftKnee;
    }

    public ModelDataBean setSinLeftKnee(double sinLeftKnee) {
        this.sinLeftKnee = sinLeftKnee;
        return this;
    }

    public double getSinLeftAnkle() {
        return sinLeftAnkle;
    }

    public ModelDataBean setSinLeftAnkle(double sinLeftAnkle) {
        this.sinLeftAnkle = sinLeftAnkle;
        return this;
    }

    public int getId() {
        return id;
    }

    public ModelDataBean setId(int id) {
        this.id = id;
        return this;
    }
}
