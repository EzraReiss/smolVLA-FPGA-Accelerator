#pragma once

#include <cstdint>

namespace smolvla {
    
    // === Type Definitions ===
    using size_t = std::size_t; // Standard size type (for shape)
    
    using float32_t = float;    // 32-bit floating point
    using bfloat16_t = float;   // 16-bit bfloat16 floating point (TODO: implement proper bfloat16 type)
    
    // === Model Dimension Constants ===
    constexpr size_t PATCH_EMBED_DIM_SMALL = 3;
    constexpr size_t PATCH_EMBED_DIM_LARGE = 16;
    constexpr size_t ACTION_STATE_DIM = 32;
    constexpr size_t ATTENTION_PROJ_DIM = 320;
    constexpr size_t HIDDEN_DIM = 720;
    constexpr size_t VISION_HIDDEN_DIM = 768;
    constexpr size_t TEXT_EMBED_DIM = 960;
    constexpr size_t VISION_POS_EMBED_DIM = 1024;
    constexpr size_t ACTION_TIME_MLP_IN_DIM = 1440;
    constexpr size_t EXPERT_FFN_DIM = 2048;
    constexpr size_t TEXT_FFN_DIM = 2560;
    constexpr size_t VISION_FFN_DIM = 3072;
    constexpr size_t CONNECTOR_PROJ_DIM = 12208;
    constexpr size_t VISION_CONNECTOR_DIM = 12288;
    constexpr size_t TOKEN_EMBED_DIM = 49280;
    
    // === Model Weight Declarations ===
    namespace model {
        namespace action_in_proj {
            const float32_t bias[HIDDEN_DIM];
            const float32_t weight[HIDDEN_DIM][ACTION_STATE_DIM];
        } // namespace action_in_proj

        namespace action_out_proj {
            const float32_t bias[ACTION_STATE_DIM];
            const float32_t weight[ACTION_STATE_DIM][HIDDEN_DIM];
        } // namespace action_out_proj

        namespace action_time_mlp_in {
            const float32_t bias[HIDDEN_DIM];
            const float32_t weight[HIDDEN_DIM][ACTION_TIME_MLP_IN_DIM];
        } // namespace action_time_mlp_in

        namespace action_time_mlp_out {
            const float32_t bias[HIDDEN_DIM];
            const float32_t weight[HIDDEN_DIM][HIDDEN_DIM];
        } // namespace action_time_mlp_out

        namespace state_proj {
            const float32_t bias[TEXT_EMBED_DIM];
            const float32_t weight[TEXT_EMBED_DIM][ACTION_STATE_DIM];
        } // namespace state_proj

        namespace vlm_with_expert {
            namespace lm_expert {
                namespace layers {
                    namespace _1 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _1

                    namespace _11 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _11

                    namespace _13 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _13

                    namespace _15 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _15

                    namespace _3 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _3

                    namespace _5 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _5

                    namespace _7 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _7

                    namespace _9 {
                        namespace self_attn {
                            namespace k_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace k_proj

                            namespace v_proj {
                                const float32_t weight[ATTENTION_PROJ_DIM][ATTENTION_PROJ_DIM];
                            } // namespace v_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                        } // namespace self_attn

                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                    } // namespace _9

                    namespace _0 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _0

                    namespace _10 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _10

                    namespace _12 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _12

                    namespace _14 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _14

                    namespace _2 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _2

                    namespace _4 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _4

                    namespace _6 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _6

                    namespace _8 {
                        namespace input_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace input_layernorm

                        namespace mlp {
                            namespace down_proj {
                                const bfloat16_t weight[HIDDEN_DIM][EXPERT_FFN_DIM];
                            } // namespace down_proj

                            namespace gate_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace gate_proj

                            namespace up_proj {
                                const bfloat16_t weight[EXPERT_FFN_DIM][HIDDEN_DIM];
                            } // namespace up_proj

                        } // namespace mlp

                        namespace post_attention_layernorm {
                            const bfloat16_t weight[HIDDEN_DIM];
                        } // namespace post_attention_layernorm

                        namespace self_attn {
                            namespace k_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace k_proj

                            namespace o_proj {
                                const bfloat16_t weight[HIDDEN_DIM][TEXT_EMBED_DIM];
                            } // namespace o_proj

                            namespace q_proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][HIDDEN_DIM];
                            } // namespace q_proj

                            namespace v_proj {
                                const bfloat16_t weight[ATTENTION_PROJ_DIM][HIDDEN_DIM];
                            } // namespace v_proj

                        } // namespace self_attn

                    } // namespace _8

                } // namespace layers

                namespace norm {
                    const bfloat16_t weight[HIDDEN_DIM];
                } // namespace norm

            } // namespace lm_expert

            namespace vlm {
                namespace lm_head {
                    const bfloat16_t weight[TOKEN_EMBED_DIM][TEXT_EMBED_DIM];
                } // namespace lm_head

                namespace model {
                    namespace connector {
                        namespace modality_projection {
                            namespace proj {
                                const bfloat16_t weight[TEXT_EMBED_DIM][VISION_CONNECTOR_DIM];
                            } // namespace proj

                        } // namespace modality_projection

                    } // namespace connector

                    namespace text_model {
                        namespace embed_tokens {
                            const bfloat16_t weight[TOKEN_EMBED_DIM][TEXT_EMBED_DIM];
                        } // namespace embed_tokens

                        namespace layers {
                            namespace _0 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _0

                            namespace _1 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _1

                            namespace _10 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _10

                            namespace _11 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _11

                            namespace _12 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _12

                            namespace _13 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _13

                            namespace _14 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _14

                            namespace _15 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _15

                            namespace _2 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _2

                            namespace _3 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _3

                            namespace _4 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _4

                            namespace _5 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _5

                            namespace _6 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _6

                            namespace _7 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _7

                            namespace _8 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _8

                            namespace _9 {
                                namespace input_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace input_layernorm

                                namespace mlp {
                                    namespace down_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_FFN_DIM];
                                    } // namespace down_proj

                                    namespace gate_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace gate_proj

                                    namespace up_proj {
                                        const bfloat16_t weight[TEXT_FFN_DIM][TEXT_EMBED_DIM];
                                    } // namespace up_proj

                                } // namespace mlp

                                namespace post_attention_layernorm {
                                    const bfloat16_t weight[TEXT_EMBED_DIM];
                                } // namespace post_attention_layernorm

                                namespace self_attn {
                                    namespace k_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace k_proj

                                    namespace o_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace o_proj

                                    namespace q_proj {
                                        const bfloat16_t weight[TEXT_EMBED_DIM][TEXT_EMBED_DIM];
                                    } // namespace q_proj

                                    namespace v_proj {
                                        const bfloat16_t weight[ATTENTION_PROJ_DIM][TEXT_EMBED_DIM];
                                    } // namespace v_proj

                                } // namespace self_attn

                            } // namespace _9

                        } // namespace layers

                        namespace norm {
                            const bfloat16_t weight[TEXT_EMBED_DIM];
                        } // namespace norm

                    } // namespace text_model

                    namespace vision_model {
                        namespace embeddings {
                            namespace patch_embedding {
                                const bfloat16_t bias[VISION_HIDDEN_DIM];
                                const bfloat16_t weight[VISION_HIDDEN_DIM][PATCH_EMBED_DIM_SMALL][PATCH_EMBED_DIM_LARGE][PATCH_EMBED_DIM_LARGE];
                            } // namespace patch_embedding

                            namespace position_embedding {
                                const bfloat16_t weight[VISION_POS_EMBED_DIM][VISION_HIDDEN_DIM];
                            } // namespace position_embedding

                        } // namespace embeddings

                        namespace encoder {
                            namespace layers {
                                namespace _0 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _0

                                namespace _1 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _1

                                namespace _10 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _10

                                namespace _11 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _11

                                namespace _2 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _2

                                namespace _3 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _3

                                namespace _4 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _4

                                namespace _5 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _5

                                namespace _6 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _6

                                namespace _7 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _7

                                namespace _8 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _8

                                namespace _9 {
                                    namespace layer_norm1 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm1

                                    namespace layer_norm2 {
                                        const bfloat16_t bias[VISION_HIDDEN_DIM];
                                        const bfloat16_t weight[VISION_HIDDEN_DIM];
                                    } // namespace layer_norm2

                                    namespace mlp {
                                        namespace fc1 {
                                            const bfloat16_t bias[VISION_FFN_DIM];
                                            const bfloat16_t weight[VISION_FFN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace fc1

                                        namespace fc2 {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_FFN_DIM];
                                        } // namespace fc2

                                    } // namespace mlp

                                    namespace self_attn {
                                        namespace k_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace k_proj

                                        namespace out_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace out_proj

                                        namespace q_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace q_proj

                                        namespace v_proj {
                                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                                            const bfloat16_t weight[VISION_HIDDEN_DIM][VISION_HIDDEN_DIM];
                                        } // namespace v_proj

                                    } // namespace self_attn

                                } // namespace _9

                            } // namespace layers

                        } // namespace encoder

                        namespace post_layernorm {
                            const bfloat16_t bias[VISION_HIDDEN_DIM];
                            const bfloat16_t weight[VISION_HIDDEN_DIM];
                        } // namespace post_layernorm

                    } // namespace vision_model

                } // namespace model

            } // namespace vlm

        } // namespace vlm_with_expert

    } // namespace model

} // namespace smolvla

