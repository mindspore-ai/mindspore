/*
 * Copyright 2023 Huawei Technologies Co., Ltd
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

package com.mindspore.micro;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class MicroTest {
    // test model: mindspore/lite/test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms
    private static final int kMSDataTypeNumberTypeFloat32 = 43;
    // lenet input shape = [2, 32, 32, 1]
    private static final int lenetInputSize = 2 * 32 * 32;
    // lenetInputSize * dataByteSize(4)
    private static final int lenetInputByteSize = lenetInputSize * 4;
    // lenet output shape = [2, 10]
    private static final int lenetOutputByteSize = 2 * 10 * 4;
    private static final float predictResultOfInputAllZeros = -2.4250138E-4F;
    private static final float eps = 1E-4F;
    private static final String weightFile = "build/lenet/src/model0/net0.bin";

    static {
        System.loadLibrary("micro_jni");
    }

    @Test
    public void shouldBuildAndFreeSuccessWhenGivenWeightFileAndInitContext() {
        MSContext context = new MSContext();
        context.init();
        Model model = new Model();

        boolean isSuccess = model.build(weightFile, context);
        assertTrue(isSuccess);

        context.free();
        model.free();
    }

    @Test
    public void shouldBuildFailedWhenNotInitContext() {
        MSContext context = new MSContext();
        Model model = new Model();

        boolean isSuccess = model.build(weightFile, context);
        assertFalse(isSuccess);

        context.free();
        model.free();
    }

    @Test
    public void shouldBuildFailedWhenGivenWrongWeightFilePath() {
        MSContext context = new MSContext();
        Model model = new Model();
        context.init();

        boolean isSuccess = model.build("", context);
        assertFalse(isSuccess);

        context.free();
        model.free();
    }

    @Test
    public void shouldSetInputsRightUsingSetBufferDataAndPredictCorrect() {
        MSContext context = new MSContext();
        Model model = new Model();
        context.init();
        model.build(weightFile, context);
        context.free();

        List<MSTensor> inputs = model.getInputs();
        assertEquals(1, inputs.size());
        assertEquals(kMSDataTypeNumberTypeFloat32, inputs.get(0).getDataType());
        assertEquals(lenetInputByteSize, inputs.get(0).size());

        boolean isSuccess;
        for (MSTensor input : inputs) {
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect((int) input.size());
            inputBuffer.order(ByteOrder.nativeOrder());
            for (int i = 0; i < lenetInputByteSize; i++) {
                inputBuffer.put((byte) 0);
            }
            isSuccess = input.setData(inputBuffer);
            assertTrue(isSuccess);
        }

        isSuccess = model.predict();
        assertTrue(isSuccess);

        List<MSTensor> outputs = model.getOutputs();
        assertEquals(1, outputs.size());
        assertEquals(kMSDataTypeNumberTypeFloat32, outputs.get(0).getDataType());
        assertEquals(lenetOutputByteSize, outputs.get(0).size());
        assertEquals(outputs.get(0).getFloatData()[0], predictResultOfInputAllZeros, eps);

        model.free();
    }

    @Test
    public void shouldSetInputsRightUsingSetFloatData() {
        MSContext context = new MSContext();
        Model model = new Model();
        context.init();
        model.build(weightFile, context);
        context.free();

        MSTensor floatInput = model.getInputs().get(0);
        float[] inputsData = new float[lenetInputSize];
        for (int i = 0; i < lenetInputSize; i++) {
            inputsData[i] = (float) i;
        }
        boolean isSuccess = floatInput.setData(inputsData);

        assertTrue(isSuccess);
        assertArrayEquals(floatInput.getFloatData(), inputsData, Float.MIN_VALUE);

        model.free();
    }

    @Test
    public void shouldPredictFailedWhenNotSetInputs() {
        MSContext context = new MSContext();
        Model model = new Model();
        context.init();
        model.build(weightFile, context);
        context.free();

        boolean isSuccess = model.predict();
        assertFalse(isSuccess);

        model.free();
    }
}
