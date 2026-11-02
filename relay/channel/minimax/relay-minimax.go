package minimax

import (
	"fmt"
	"strings"

	channelconstant "github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/constant"
	"github.com/QuantumNous/new-api/relaykit/types"
)

func GetRequestURL(info *relaycommon.RelayInfo) (string, error) {
	baseUrl := info.ChannelBaseUrl
	if baseUrl == "" {
		baseUrl = channelconstant.GetChannelBaseURL(channelconstant.ChannelTypeMiniMax)
	}
	// Accept both provider roots and SDK base URLs without duplicating prefixes.
	baseUrl = strings.TrimRight(baseUrl, "/")
	baseUrl = strings.TrimSuffix(baseUrl, "/v1")
	baseUrl = strings.TrimSuffix(baseUrl, "/anthropic")
	switch info.RelayFormat {
	case types.RelayFormatClaude:
		return fmt.Sprintf("%s/anthropic/v1/messages", baseUrl), nil
	case types.RelayFormatOpenAIResponses:
		return fmt.Sprintf("%s/v1/responses", baseUrl), nil
	default:
		switch info.RelayMode {
		case constant.RelayModeChatCompletions:
			model := info.UpstreamModelName
			if model == "" {
				model = info.OriginModelName
			}
			if strings.HasPrefix(model, "abab") {
				return fmt.Sprintf("%s/v1/text/chatcompletion_v2", baseUrl), nil
			}
			return fmt.Sprintf("%s/v1/chat/completions", baseUrl), nil
		case constant.RelayModeImagesGenerations:
			return fmt.Sprintf("%s/v1/image_generation", baseUrl), nil
		case constant.RelayModeAudioSpeech:
			return fmt.Sprintf("%s/v1/t2a_v2", baseUrl), nil
		default:
			return "", fmt.Errorf("unsupported relay mode: %d", info.RelayMode)
		}
	}
}
