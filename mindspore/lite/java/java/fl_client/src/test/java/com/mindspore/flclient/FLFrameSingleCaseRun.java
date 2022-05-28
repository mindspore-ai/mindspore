package com.mindspore.flclient;

import com.mindspore.flclient.model.Callback;
import mockit.*;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**   
 * Read test cases from case define file, just run the specified case.
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLFrameSingleCaseRun {
    static FLMockServer webServer =  new FLMockServer();
    static public List<Object[]> testCases;
    static String utBasePath;
    int sleepCallTimes = 0;

    @BeforeClass
    public static void runBeforeAllTheTest() throws IOException {
        utBasePath = System.getenv("MS_FL_UT_BASE_PATH");
        String projectPath = System.getProperty("user.dir");
        if (utBasePath == null || utBasePath.isEmpty()) {
            utBasePath = projectPath;
        }
        String caseFilePath = utBasePath + "/test_case/fl_frame_ut_case_infer.json";
        FLFrameTestCaseParser caseParser = new FLFrameTestCaseParser(caseFilePath);
        testCases = caseParser.Parser();

        // If MS_CATCH_MSG is set must have real server node, and close the mock server
        if (System.getenv("MS_CATCH_MSG") == null) {
            webServer = new FLMockServer();
            webServer.run(6668);
        }
    }

    @AfterClass
    public static void runAfterAllTheTest() throws Exception {
        webServer.stop();
    }

    @Test
    public void testMain() {
        // mock
        MockUp<Common> mockCommon = new MockUp<Common>() {
            @Mock
            public void sleep(long millis) {
                sleepCallTimes++;
                return;
            }
        };

        FLFrameTestCase flTestCase = null;
        for(Object[] arrObj : testCases){
            String caseName = (String) arrObj[0];
            if(caseName.equals("New_Frame_Tag_Infer")){
                flTestCase = (FLFrameTestCase)arrObj[1];
            }
        }
        assertNotEquals(flTestCase, null);

        if (System.getenv("MS_CATCH_MSG") != null) {
            FLCommunication.setMsgDumpFlg(true);
            FLCommunication.setMsgDumpPath(utBasePath + "/test_data/" + flTestCase.getFlName());
        } else {
            webServer.setCaseRes(flTestCase.getHttpRes());
        }
        flTestCase.getRealPath(utBasePath);
        webServer.setCaseRes(flTestCase.getHttpRes());

        FLParameter.getInstance().setIflJobResultCallback(new FLUTResultCheck(flTestCase.getResultCode(), new int[0]));

        SyncFLJob.main(flTestCase.getParams());
    }
}