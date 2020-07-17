config = {
    # model & pickle paths
    "model_path": "models/english_to_spanish_attention_nmt.h5",
    "encoder_path": "models/encoder_model.h5",
    "decoder_path": "models/decoder_model.h5",
    "input_word_index": "pickles/input_word_index.pkl",
    "target_word_index": "pickles/target_word_index.pkl",
    # translation url
    "url": "https://api.mymemory.translated.net/get",
    # constants
    "max_length_src": 47,
    "max_length_tar": 47,
}
