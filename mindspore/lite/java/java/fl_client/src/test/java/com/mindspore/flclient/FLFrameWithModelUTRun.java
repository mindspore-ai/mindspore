package com.mindspore.flclient;

import mockit.Mock;
import mockit.MockUp;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**   
 * The Parameterized UT class for FlFrame which include model code
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
@RunWith(Parameterized.class)
public class FLFrameWithModelUTRun {
    static FLMockServer mockServer;
    static String utBasePath;
    String caseName;
    FLFrameWithModelTestCase curCase;

    public FLFrameWithModelUTRun(String caseName, FLFrameWithModelTestCase curCase) {
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
        String caseFilePath = utBasePath + "/test_case/fl_frame_with_model_ut_cases.json";
        FLFrameWithModelTestCaseParser caseParser = new FLFrameWithModelTestCaseParser(caseFilePath);
        return caseParser.Parser();
    }

    /**
     * @throws IOException
     */
    @BeforeClass
    public static void runBeforeAllTheTest() throws IOException {
        // If MS_CATCH_MSG is set must have real server node, and close the mock server
        if (System.getenv("MS_CATCH_MSG") == null) {
            mockServer = new FLMockServer();
            mockServer.run(6668);
        }
    }

    @AfterClass
    public static void runAfterAllTheTest() {
        mockServer.stop();
    }


    @Before
    public void runBeforeEachTest() {
        // config how to check
        FLParameter.getInstance().setIflJobResultCallback(new FLUTResultCheck(curCase.getResultCode(), new int[0]));
        // using real server node to catch msg for mock server.
        if (System.getenv("MS_CATCH_MSG") != null) {
            FLCommunication.setMsgDumpFlg(true);
            FLCommunication.setMsgDumpPath(utBasePath + "/test_data/" + curCase.getFlName());
        } else {
            mockServer.setCaseRes(curCase.getHttpRes());
        }
        curCase.getRealPath(utBasePath);
    }

    @After
    public void runAfterEachTest() {
    }

    @Test
    public void runTestCase() {
        // mock
        MockUp<Common> mockCommon = new MockUp<Common>() {
            @Mock
            public void sleep(long millis) {
                return;
            }
        };
        SyncFLJob.main(curCase.getParams());
    }
}
