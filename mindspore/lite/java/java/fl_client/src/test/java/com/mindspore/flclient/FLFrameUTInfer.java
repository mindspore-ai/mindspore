package com.mindspore.flclient;

import mockit.*;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.util.List;

/**
 * The Parameterized UT class for FlFrame which not include model code
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
@RunWith(Parameterized.class)
public class FLFrameUTInfer {
    static FLMockServer mockServer;
    static String utBasePath;
    String caseName;
    FLFrameTestCase curCase;

    public FLFrameUTInfer(String caseName, FLFrameTestCase curCase) {
        this.caseName = caseName;
        this.curCase = curCase;
    }

    @Parameterized.Parameters(name = "{index}: {0}")
    public static List<Object[]> cases() throws IOException {
        utBasePath = System.getenv("MS_FL_UT_BASE_PATH");
        String projectPath = System.getProperty("user.dir");
        if (utBasePath == null || utBasePath.isEmpty()) {
            utBasePath = projectPath;
        }
        String caseFilePath = utBasePath + "/test_case/fl_frame_ut_case_infer.json";
        FLFrameTestCaseParser caseParser = new FLFrameTestCaseParser(caseFilePath);
        return caseParser.Parser();
    }

    @Before
    public void runBeforeEachTest(){
        curCase.getRealPath(utBasePath);
    }
    @After
    public void runAfterEachTest(){
    }

    @Test
    public void runTestCase() {
        SyncFLJob.main(curCase.getParams());
    }
}
