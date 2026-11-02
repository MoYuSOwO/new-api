package minimax

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	relayconstant "github.com/QuantumNous/new-api/relay/constant"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/setting/operation_setting"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetRequestURLForImageGeneration(t *testing.T) {
	t.Parallel()

	info := &relaycommon.RelayInfo{
		RelayMode: relayconstant.RelayModeImagesGenerations,
		ChannelMeta: &relaycommon.ChannelMeta{
			ChannelBaseUrl: "https://api.minimax.chat",
		},
	}

	got, err := GetRequestURL(info)
	if err != nil {
		t.Fatalf("GetRequestURL returned error: %v", err)
	}

	want := "https://api.minimax.chat/v1/image_generation"
	if got != want {
		t.Fatalf("GetRequestURL() = %q, want %q", got, want)
	}
}

func TestConvertImageRequest(t *testing.T) {
	t.Parallel()

	adaptor := &Adaptor{}
	info := &relaycommon.RelayInfo{
		RelayMode:       relayconstant.RelayModeImagesGenerations,
		OriginModelName: "image-01",
	}
	request := dto.ImageRequest{
		Model:          "image-01",
		Prompt:         "a red fox in snowfall",
		Size:           "1536x1024",
		ResponseFormat: "url",
		N:              uintPtr(2),
	}

	got, err := adaptor.ConvertImageRequest(gin.CreateTestContextOnly(httptest.NewRecorder(), gin.New()), info, request)
	if err != nil {
		t.Fatalf("ConvertImageRequest returned error: %v", err)
	}

	body, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("json.Marshal returned error: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("json.Unmarshal returned error: %v", err)
	}

	if payload["model"] != "image-01" {
		t.Fatalf("model = %#v, want %q", payload["model"], "image-01")
	}
	if payload["prompt"] != request.Prompt {
		t.Fatalf("prompt = %#v, want %q", payload["prompt"], request.Prompt)
	}
	if payload["n"] != float64(2) {
		t.Fatalf("n = %#v, want 2", payload["n"])
	}
	if payload["aspect_ratio"] != "3:2" {
		t.Fatalf("aspect_ratio = %#v, want %q", payload["aspect_ratio"], "3:2")
	}
	if payload["response_format"] != "url" {
		t.Fatalf("response_format = %#v, want %q", payload["response_format"], "url")
	}
}

func TestDoResponseForImageGeneration(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)

	info := &relaycommon.RelayInfo{
		RelayMode: relayconstant.RelayModeImagesGenerations,
		StartTime: time.Unix(1700000000, 0),
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       httptest.NewRecorder().Result().Body,
	}
	resp.Body = ioNopCloser(`{"data":{"image_urls":["https://example.com/minimax.png"]}}`)

	adaptor := &Adaptor{}
	usage, err := adaptor.DoResponse(c, resp, info)
	if err != nil {
		t.Fatalf("DoResponse returned error: %v", err)
	}
	if usage == nil {
		t.Fatalf("DoResponse returned nil usage")
	}

	body := recorder.Body.String()
	if !strings.Contains(body, `"url":"https://example.com/minimax.png"`) {
		t.Fatalf("response body = %s, want OpenAI image response with image URL", body)
	}
	if strings.Contains(body, `"image_urls"`) {
		t.Fatalf("response body = %s, should not expose raw MiniMax image_urls payload", body)
	}
}

type nopReadCloser struct {
	*strings.Reader
}

func (n nopReadCloser) Close() error {
	return nil
}

func ioNopCloser(body string) nopReadCloser {
	return nopReadCloser{Reader: strings.NewReader(body)}
}

func uintPtr(v uint) *uint {
	return &v
}

func TestNativeTextRequestURLs(t *testing.T) {
	for _, test := range []struct {
		name, base, model, want string
		format                  types.RelayFormat
		mode                    int
	}{
		{"default-chat", "", "MiniMax-M3.1-Flash", "https://api.minimax.io/v1/chat/completions", types.RelayFormatOpenAI, relayconstant.RelayModeChatCompletions},
		{"default-messages", "", "MiniMax-M3", "https://api.minimax.io/anthropic/v1/messages", types.RelayFormatClaude, relayconstant.RelayModeChatCompletions},
		{"default-responses", "", "MiniMax-M3.1-Flash", "https://api.minimax.io/v1/responses", types.RelayFormatOpenAIResponses, relayconstant.RelayModeResponses},
		{"regional-chat", "https://api.minimaxi.com/", "MiniMax-M2.7", "https://api.minimaxi.com/v1/chat/completions", types.RelayFormatOpenAI, relayconstant.RelayModeChatCompletions},
		{"v1-base", "https://api.minimax.io/v1/", "MiniMax-M3", "https://api.minimax.io/v1/responses", types.RelayFormatOpenAIResponses, relayconstant.RelayModeResponses},
		{"anthropic-base", "https://api.minimax.io/anthropic", "MiniMax-M3", "https://api.minimax.io/anthropic/v1/messages", types.RelayFormatClaude, relayconstant.RelayModeChatCompletions},
		{"anthropic-v1-base", "https://api.minimax.io/anthropic/v1/", "MiniMax-M3", "https://api.minimax.io/anthropic/v1/messages", types.RelayFormatClaude, relayconstant.RelayModeChatCompletions},
		{"gateway-prefix", "https://gateway.example/minimax/v1", "MiniMax-M3", "https://gateway.example/minimax/v1/chat/completions", types.RelayFormatOpenAI, relayconstant.RelayModeChatCompletions},
		{"legacy-abab", "https://api.minimax.chat", "abab6.5-chat", "https://api.minimax.chat/v1/text/chatcompletion_v2", types.RelayFormatOpenAI, relayconstant.RelayModeChatCompletions},
		{"speech", "https://api.minimax.io/", "speech-2.8-hd", "https://api.minimax.io/v1/t2a_v2", types.RelayFormatOpenAI, relayconstant.RelayModeAudioSpeech},
		{"image", "https://api.minimax.io/v1", "image-01", "https://api.minimax.io/v1/image_generation", types.RelayFormatOpenAI, relayconstant.RelayModeImagesGenerations},
	} {
		t.Run(test.name, func(t *testing.T) {
			info := &relaycommon.RelayInfo{RelayFormat: test.format, RelayMode: test.mode, OriginModelName: "client-alias", ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: test.base, UpstreamModelName: test.model}}
			url, err := GetRequestURL(info)
			require.NoError(t, err)
			assert.Equal(t, test.want, url)
		})
	}
}

// Exercise the full adaptor transport against a fake provider. Requests retain
// native thinking/tool history; returned usage must come from the provider.
func TestNativeTextProtocols(t *testing.T) {
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() { constant.StreamingTimeout = oldTimeout })
	operation_setting.SetToolPriceForTest("echo", 5)
	t.Cleanup(func() { operation_setting.DeleteToolPriceForTest("echo") })
	for _, format := range []types.RelayFormat{types.RelayFormatOpenAI, types.RelayFormatClaude, types.RelayFormatOpenAIResponses} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", format, stream), func(t *testing.T) {
				requestJSON, responseBody, path := nativeTextFixture(format, stream)
				var expected map[string]any
				require.NoError(t, common.UnmarshalJsonStr(requestJSON, &expected))
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, path, r.URL.Path)
					var actual map[string]any
					if !assert.NoError(t, common.DecodeJson(r.Body, &actual)) {
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					for key, value := range expected {
						assert.Equal(t, value, actual[key], "request field %s", key)
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
					} else {
						w.Header().Set("Content-Type", "application/json")
					}
					_, _ = io.WriteString(w, responseBody)
				}))
				defer server.Close()
				recorder := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(recorder)
				c.Request = httptest.NewRequest(http.MethodPost, path, strings.NewReader(requestJSON))
				c.Request.Header.Set("Content-Type", "application/json")
				info := &relaycommon.RelayInfo{RelayFormat: format, RelayMode: relayconstant.RelayModeChatCompletions, IsStream: stream, DisablePing: true, ShouldIncludeUsage: true, OriginModelName: "MiniMax-M3.1-Flash", ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeMiniMax, ChannelBaseUrl: server.URL, UpstreamModelName: "MiniMax-M3.1-Flash", ApiKey: "test-key"}}
				info.EnsureClaudeConvertInfo()
				adaptor := &Adaptor{}
				var converted any
				var err error
				switch format {
				case types.RelayFormatClaude:
					var request dto.ClaudeRequest
					require.NoError(t, common.UnmarshalJsonStr(requestJSON, &request))
					converted, err = adaptor.ConvertClaudeRequest(c, info, &request)
				case types.RelayFormatOpenAIResponses:
					info.RelayMode = relayconstant.RelayModeResponses
					var request dto.OpenAIResponsesRequest
					require.NoError(t, common.UnmarshalJsonStr(requestJSON, &request))
					converted, err = adaptor.ConvertOpenAIResponsesRequest(c, info, request)
				default:
					var request dto.GeneralOpenAIRequest
					require.NoError(t, common.UnmarshalJsonStr(requestJSON, &request))
					converted, err = adaptor.ConvertOpenAIRequest(c, info, &request)
				}
				require.NoError(t, err)
				body, err := common.Marshal(converted)
				require.NoError(t, err)
				response, err := adaptor.DoRequest(c, info, strings.NewReader(string(body)))
				require.NoError(t, err)
				upstream, ok := response.(*http.Response)
				require.True(t, ok)
				require.Equal(t, http.StatusOK, upstream.StatusCode)
				usageValue, apiErr := adaptor.DoResponse(c, upstream, info)
				require.Nil(t, apiErr)
				usage, ok := usageValue.(*dto.Usage)
				require.True(t, ok)
				require.NotNil(t, usage)
				prompt := 12
				if format == types.RelayFormatClaude {
					prompt = 8
				}
				assert.Equal(t, prompt, usage.PromptTokens)
				assert.Equal(t, 3, usage.CompletionTokens)
				assert.Equal(t, 4, usage.PromptTokensDetails.CachedTokens)
				require.NotNil(t, info.ResponsesUsageInfo)
				require.Contains(t, info.ResponsesUsageInfo.BuiltInTools, "echo")
				assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools["echo"].CallCount)
				output := recorder.Body.String()
				assert.Contains(t, output, "call_native_long_id")
				assert.Contains(t, output, "echo")
				assert.Contains(t, output, "consider carefully")
				if stream {
					assert.Contains(t, recorder.Header().Get("Content-Type"), "text/event-stream")
					terminal := "[DONE]"
					if format == types.RelayFormatClaude {
						terminal = "event: message_stop"
					}
					if format == types.RelayFormatOpenAIResponses {
						terminal = "event: response.completed"
					}
					assert.Contains(t, output, terminal)
				} else {
					assert.JSONEq(t, responseBody, output)
				}
			})
		}
	}
}

func nativeTextFixture(format types.RelayFormat, stream bool) (request, response, path string) {
	switch format {
	case types.RelayFormatClaude:
		path = "/anthropic/v1/messages"
		request = `{"model":"MiniMax-M3.1-Flash","max_tokens":64,"thinking":{"type":"adaptive"},"output_config":{"effort":"max"},"messages":[{"role":"user","content":"echo"},{"role":"assistant","content":[{"type":"thinking","thinking":"history","signature":"sig"},{"type":"tool_use","id":"old_call","name":"echo","input":{"value":7}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"old_call","content":"7"}]}],"tools":[{"name":"echo","input_schema":{"type":"object","properties":{"value":{"type":"integer"}}}}]`
		response = `{"id":"msg_native","type":"message","role":"assistant","model":"MiniMax-M3.1-Flash","content":[{"type":"thinking","thinking":"consider carefully","signature":"sig"},{"type":"tool_use","id":"call_native_long_id","name":"echo","input":{"value":7}}],"stop_reason":"tool_use","usage":{"input_tokens":8,"output_tokens":3,"cache_read_input_tokens":4}}`
		if stream {
			response = strings.Join([]string{
				"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_native\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M3.1-Flash\",\"content\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":0,\"cache_read_input_tokens\":4}}}",
				`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"consider carefully"}}`,
				`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
				`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_native_long_id","name":"echo","input":{}}}`,
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"value\":"}}`,
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"7}"}}`,
				`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":1}`,
				`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
				`event: message_stop` + "\n" + `data: {"type":"message_stop"}`, "",
			}, "\n\n")
		}
	case types.RelayFormatOpenAIResponses:
		path = "/v1/responses"
		request = `{"model":"MiniMax-M3.1-Flash","max_output_tokens":64,"reasoning":{"effort":"max"},"store":false,"parallel_tool_calls":false,"input":[{"role":"user","content":"echo"},{"type":"function_call","call_id":"old_call","name":"echo","arguments":"{\"value\":7}"},{"type":"function_call_output","call_id":"old_call","output":"7"}],"tools":[{"type":"function","name":"echo","parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}]`
		response = `{"id":"resp_native","object":"response","status":"completed","created_at":1710000000,"model":"MiniMax-M3.1-Flash","output":[{"id":"rs_1","type":"reasoning","content":[{"type":"reasoning_text","text":"consider carefully"}]},{"id":"fc_1","type":"function_call","call_id":"call_native_long_id","name":"echo","arguments":"{\"value\":7}"}],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":4}}}`
		if stream {
			response = strings.Join([]string{
				`data: {"type":"response.created","response":{"id":"resp_native","object":"response","status":"in_progress","model":"MiniMax-M3.1-Flash","output":[]}}`,
				`data: {"type":"response.reasoning_text.delta","item_id":"rs_1","delta":"consider carefully"}`,
				`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_native_long_id","name":"echo","arguments":""}}`,
				`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"value\":"}`,
				`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"7}"}`,
				`data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_native_long_id","name":"echo","arguments":"{\"value\":7}"}}`,
				`data: {"type":"response.completed","response":` + response + `}`, "",
			}, "\n\n")
		}
	default:
		path = "/v1/chat/completions"
		request = `{"model":"MiniMax-M3.1-Flash","max_completion_tokens":64,"reasoning_effort":"max","thinking":{"type":"adaptive"},"parallel_tool_calls":false,"messages":[{"role":"user","content":"echo"},{"role":"assistant","content":"","reasoning_content":"history","tool_calls":[{"id":"old_call","type":"function","function":{"name":"echo","arguments":"{\"value\":7}"}}]},{"role":"tool","tool_call_id":"old_call","content":"7"}],"tools":[{"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}}]`
		response = `{"id":"chat_native","object":"chat.completion","created":1710000000,"model":"MiniMax-M3.1-Flash","choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"consider carefully","tool_calls":[{"id":"call_native_long_id","type":"function","function":{"name":"echo","arguments":"{\"value\":7}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}`
		if stream {
			response = strings.Join([]string{
				`data: {"id":"chat_native","model":"MiniMax-M3.1-Flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"consider carefully","tool_calls":[{"index":0,"id":"call_native_long_id","type":"function","function":{"name":"echo","arguments":""}}]}}]}`,
				`data: {"id":"chat_native","model":"MiniMax-M3.1-Flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"value\":"}}]}}]}`,
				`data: {"id":"chat_native","model":"MiniMax-M3.1-Flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"7}"}}]},"finish_reason":"tool_calls"}]}`,
				`data: {"id":"chat_native","model":"MiniMax-M3.1-Flash","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}`,
				`data: [DONE]`, "",
			}, "\n\n")
		}
	}
	request = strings.TrimSuffix(request, "}") + fmt.Sprintf(`,"stream":%t}`, stream)
	return
}
