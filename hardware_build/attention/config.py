#cross attention configuration
# runtime / scheduling

class CrossAttentionConfig:
    LENGTH_OF_ACTION_CHUNK = 50          # aka chunk_size / n_action_steps = 50
    NUM_STEPS             = 10           # flow-matching solver steps


    EXPERT_WIDTH_MULT     = 0.75         # action expert width vs VLM width

    # action expert channel sizes (from weights)
    ACTION_HIDDEN_SIZE    = 720          # token width inside expert
    Q_PROJ_OUT_DIM        = 960          # implies 12 Q heads at head_dim=80
    KV_DIM                = 320          # CA K/V live in 320-dim per token
    NUM_Q_HEADS           = 12
    NUM_KV_HEADS          = 4
    HEAD_DIM              = 80           # so 960=12*80 and 320=4*80
    O_PROJ_IN_DIM         = 960
    O_PROJ_OUT_DIM        = 720

    # text & observation packing (defaults on the hub)
    TOKENIZER_MAX_LENGTH  = 48
    N_OBS_STEPS           = 1
    NUM_CAMERAS           = 3
    VIS_TOKENS_PER_FRAME  = 64      
    STATE_TOKEN_COUNT     = 1

    # convenience (derived, not fixed by weights)
    DEFAULT_Tf            = 64*NUM_CAMERAS*N_OBS_STEPS + TOKENIZER_MAX_LENGTH + STATE_TOKEN_COUNT


class VLMAttentionConfig:
    NUM_HEADS = 12
    HIDDEN_DIM = 768
    SINGLE_HEAD_DIM = HIDDEN_DIM // NUM_HEADS
    NUM_LAYERS = 12
    NUM_TOKENS = 1024


