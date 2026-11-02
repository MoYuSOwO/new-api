package mistral

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/setting/operation_setting"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCrossProtocolToolRequests(t *testing.T) {
	for _, format := range []types.RelayFormat{types.RelayFormatClaude, types.RelayFormatOpenAIResponses} {
		t.Run(string(format), func(t *testing.T) {
			info := &relaycommon.RelayInfo{RelayFormat: format, ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeMistral, ChannelBaseUrl: "https://api.mistral.ai"}}
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			var converted any
			var err error
			if format == types.RelayFormatClaude {
				info.RequestURLPath = "/v1/messages?beta=true"
				var request dto.ClaudeRequest
				require.NoError(t, common.UnmarshalJsonStr(`{"model":"devstral-test","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"run echo"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_long_original_id","name":"echo","input":{"value":7}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_long_original_id","content":"7"}]}],"tools":[{"name":"echo","input_schema":{"type":"object","properties":{"value":{"type":"integer"}}}}]}`, &request))
				request.StopSequences = []string{"END", "STOP"}
				converted, err = (&Adaptor{}).ConvertClaudeRequest(c, info, &request)
			} else {
				info.RequestURLPath = "/v1/responses"
				var request dto.OpenAIResponsesRequest
				require.NoError(t, common.UnmarshalJsonStr(`{"model":"devstral-test","max_output_tokens":64,"temperature":0,"store":false,"input":[{"role":"user","content":"run echo"},{"type":"function_call","call_id":"call_long_original_id","name":"echo","arguments":"{\"value\":7}"},{"type":"function_call_output","call_id":"call_long_original_id","output":"7"}],"tools":[{"type":"function","name":"echo","parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}]}`, &request))
				request.Text = []byte(`{"format":{"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"value":{"type":"integer"}}},"strict":true}}`)
				request.ParallelToolCalls = []byte(`false`)
				converted, err = (&Adaptor{}).ConvertOpenAIResponsesRequest(c, info, request)
			}
			require.NoError(t, err)
			request, ok := converted.(*dto.GeneralOpenAIRequest)
			require.True(t, ok)
			assert.Equal(t, "devstral-test", request.Model)
			require.NotNil(t, request.MaxTokens)
			assert.EqualValues(t, 64, *request.MaxTokens)
			require.NotNil(t, request.Temperature)
			assert.Zero(t, *request.Temperature)
			require.Len(t, request.Messages, 3)
			calls := request.Messages[1].ParseToolCalls()
			require.Len(t, calls, 1)
			assert.Regexp(t, `^[a-zA-Z0-9]{9}$`, calls[0].ID)
			assert.Equal(t, calls[0].ID, request.Messages[2].ToolCallId)
			assert.Equal(t, "echo", calls[0].Function.Name)
			assert.JSONEq(t, `{"value":7}`, calls[0].Function.Arguments)
			require.Len(t, request.Tools, 1)
			body, err := common.Marshal(request)
			require.NoError(t, err)
			assert.NotContains(t, string(body), `"store"`)
			if format == types.RelayFormatClaude {
				assert.Equal(t, []string{"END", "STOP"}, request.Stop)
			} else {
				require.NotNil(t, request.ResponseFormat)
				assert.Equal(t, "json_schema", request.ResponseFormat.Type)
				assert.JSONEq(t, `{"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"value":{"type":"integer"}}},"strict":true}`, string(request.ResponseFormat.JsonSchema))
				require.NotNil(t, request.ParallelTooCalls)
				assert.False(t, *request.ParallelTooCalls)
			}
			url, err := (&Adaptor{}).GetRequestURL(info)
			require.NoError(t, err)
			assert.Equal(t, "https://api.mistral.ai/v1/chat/completions", url)
			assert.Equal(t, format, info.RelayFormat)
		})
	}
}

func TestCrossProtocolToolResponses(t *testing.T) {
	operation_setting.SetToolPriceForTest("echo", 5)
	t.Cleanup(func() { operation_setting.DeleteToolPriceForTest("echo") })
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() { constant.StreamingTimeout = oldTimeout })
	for _, format := range []types.RelayFormat{types.RelayFormatClaude, types.RelayFormatOpenAIResponses, types.RelayFormatOpenAI} {
		for _, stream := range []bool{false, true} {
			name := string(format) + "/json"
			if stream {
				name = string(format) + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				body := `{"id":"chat-1","object":"chat.completion","created":1710000000,"model":"devstral-test","choices":[{"index":0,"message":{"role":"assistant","content":"hello","tool_calls":[{"id":"abc123XYZ","type":"function","function":{"name":"echo","arguments":"{\"value\":7}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}`
				if stream {
					body = strings.Join([]string{
						`data: {"id":"chat-1","model":"devstral-test","choices":[{"index":0,"delta":{"role":"assistant","content":"hello","tool_calls":[{"index":0,"id":"abc123XYZ","type":"function","function":{"name":"echo","arguments":""}}]}}]}`,
						`data: {"id":"chat-1","model":"devstral-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"value\":"}}]}}]}`,
						`data: {"id":"chat-1","model":"devstral-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"7}"}}]},"finish_reason":"tool_calls"}]}`,
						`data: {"id":"chat-1","model":"devstral-test","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}`,
						`data: [DONE]`, "",
					}, "\n\n")
				}
				recorder := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(recorder)
				c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
				info := &relaycommon.RelayInfo{RelayFormat: format, IsStream: stream, DisablePing: true, ShouldIncludeUsage: true, ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeMistral, UpstreamModelName: "devstral-test"}}
				info.EnsureClaudeConvertInfo()
				resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
				usageValue, apiErr := (&Adaptor{}).DoResponse(c, resp, info)
				require.Nil(t, apiErr)
				usage, ok := usageValue.(*dto.Usage)
				require.True(t, ok)
				require.NotNil(t, usage)
				assert.Equal(t, 12, usage.PromptTokens)
				assert.Equal(t, 3, usage.CompletionTokens)
				assert.Equal(t, 15, usage.TotalTokens)
				require.NotNil(t, info.ResponsesUsageInfo)
				require.Contains(t, info.ResponsesUsageInfo.BuiltInTools, "echo")
				assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools["echo"].CallCount)
				if format == types.RelayFormatOpenAIResponses {
					require.NotNil(t, usage.InputTokensDetails)
					assert.Equal(t, 4, usage.InputTokensDetails.CachedTokens)
				} else {
					assert.Equal(t, 4, usage.PromptTokensDetails.CachedTokens)
				}
				output := recorder.Body.String()
				assert.Contains(t, output, "abc123XYZ")
				assert.Contains(t, output, "echo")
				if stream {
					switch format {
					case types.RelayFormatClaude:
						assert.Contains(t, output, "event: content_block_delta")
						assert.Contains(t, output, "event: message_stop")
						assert.Contains(t, output, `"stop_reason":"tool_use"`)
					case types.RelayFormatOpenAIResponses:
						assert.Contains(t, output, "event: response.function_call_arguments.delta")
						assert.Contains(t, output, "event: response.completed")
						assert.Contains(t, output, `"arguments":"{\"value\":7}"`)
					default:
						assert.Contains(t, output, "[DONE]")
					}
				} else {
					var result map[string]any
					require.NoError(t, common.UnmarshalJsonStr(output, &result))
					switch format {
					case types.RelayFormatClaude:
						assert.Equal(t, "message", result["type"])
						assert.Equal(t, "tool_use", result["stop_reason"])
					case types.RelayFormatOpenAIResponses:
						assert.Equal(t, "response", result["object"])
						assert.Contains(t, output, `"type":"function_call"`)
					default:
						assert.Equal(t, "chat.completion", result["object"])
					}
				}
			})
		}
	}
}

func TestNativeRequestURLPreserved(t *testing.T) {
	info := &relaycommon.RelayInfo{RelayFormat: types.RelayFormatOpenAI, RequestURLPath: "/v1/fim/completions", ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeMistral, ChannelBaseUrl: "https://api.mistral.ai"}}
	url, err := (&Adaptor{}).GetRequestURL(info)
	require.NoError(t, err)
	assert.Equal(t, "https://api.mistral.ai/v1/fim/completions", url)
}

func TestNilClaudeRequestRejected(t *testing.T) {
	converted, err := (&Adaptor{}).ConvertClaudeRequest(nil, nil, nil)
	require.Error(t, err)
	assert.Nil(t, converted)
}

func TestStatefulResponsesRequestRejected(t *testing.T) {
	info := &relaycommon.RelayInfo{RelayFormat: types.RelayFormatOpenAIResponses, ChannelMeta: &relaycommon.ChannelMeta{}}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	var request dto.OpenAIResponsesRequest
	require.NoError(t, common.UnmarshalJsonStr(`{"model":"devstral-test","input":"hello","previous_response_id":"resp_previous"}`, &request))
	converted, err := (&Adaptor{}).ConvertOpenAIResponsesRequest(c, info, request)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "previous_response_id")
	assert.Nil(t, converted)
}
