use std::sync::Arc;

use crate::incremental::DecodeStream;

mod byte_level_decode;
#[macro_use]
mod error;
mod hf;
mod incremental;
mod tekken;
mod tiktoken;

pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::IncrementalDecoder;
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;

pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Convert one token string into a token ID, returning `None` if the token
    /// is not in the tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Convert one token ID into the tokenizer's raw token string.
    fn id_to_token(&self, _id: u32) -> Option<String> {
        // TODO: remove default impl and require this to be implemented by all
        // tokenizers
        None
    }

    /// Return the vocabulary size. Backends that cannot report it fall back to
    /// `usize::MAX`, an effectively unbounded value used only by test stubs.
    fn vocab_size(&self) -> usize {
        usize::MAX
    }

    /// Return whether the given token ID is special.
    fn is_special_id(&self, _token_id: u32) -> bool {
        false
    }

    /// Whether `decode` is context-independent: every token decodes to the same
    /// bytes regardless of the surrounding tokens.
    ///
    /// This holds for byte-level tokenizers but not for context-dependent
    /// decoders (e.g. Metaspace/SentencePiece), where a token's rendering
    /// depends on its position — a leading-space token decodes differently in
    /// isolation than mid-sequence. The incremental decoder only reuses a
    /// decoded suffix as the prefix seed when this is `true`; otherwise it must
    /// decode from the full prompt. Defaults to `false` (the safe choice).
    fn decode_is_context_independent(&self) -> bool {
        false
    }

    /// Create a stateful incremental decoder primed with the given prompt
    /// tokens.
    ///
    /// The prompt tokens provide left context for the first generated token;
    /// the decoder does not re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

pub type DynTokenizer = Arc<dyn Tokenizer>;
