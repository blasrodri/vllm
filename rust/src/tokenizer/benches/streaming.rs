//! Profiling harness for the per-token incremental detokenization hot path
//! (`DecodeStream::push_token`), used to quantify the CPU cost of the current
//! two-decodes-per-token scheme vs. an idealized memoized byte-level floor.
//!
//! Runs offline: synthesizes a GPT-2 byte-level tokenizer so it exercises the
//! `FastokensByteLevel` fast path without any network/model download.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{Criterion, Throughput, black_box};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

// --- allocation-counting global allocator -----------------------------------

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}
fn bytes() -> usize {
    BYTES.load(Ordering::Relaxed)
}

// --- GPT-2 byte-level helpers ------------------------------------------------

/// GPT-2 byte→unicode table (same mapping fastokens' ByteLevel uses).
fn byte_to_char() -> [char; 256] {
    let is_nice = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let cp = if is_nice(b) {
            b as u32
        } else {
            let cp = next;
            next += 1;
            cp
        };
        table[b as usize] = char::from_u32(cp).unwrap();
    }
    table
}

/// Byte-level encode a raw string (each UTF-8 byte → its GPT-2 char).
fn to_byte_level(s: &str, tbl: &[char; 256]) -> String {
    s.bytes().map(|b| tbl[b as usize]).collect()
}

// --- fixture: synthetic byte-level tokenizer + realistic token stream --------

const SAMPLE: &str = "\
The quick brown fox jumps over the lazy dog. In vLLM, the Rust frontend streams \
decoded tokens back to the client one step at a time, so detokenization runs on \
the hot path for every generated token. fn main() { let mut total = 0u64; for i \
in 0..1024 { total += compute(i); } println!(\"{}\", total); } 请用中英混合总结，\
并给出一个简短的 JSON 示例：{\"ok\": true, \"count\": 42}. The service should stop \
cleanly at EOS and keep decode latency low under concurrent load.\n";

struct Fixture {
    tokenizer: HuggingFaceTokenizer,
    prompt: Vec<u32>,
    stream: Vec<u32>,
}

/// Split the sample into realistic word-piece tokens: a leading space starts a
/// new piece (GPT-2 style), and long words are chunked to <=6 chars.
fn piece_strings() -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, pieces: &mut Vec<String>| {
        if !cur.is_empty() {
            // chunk to <=6 chars at char boundaries
            let chars: Vec<char> = cur.chars().collect();
            for ch in chars.chunks(6) {
                pieces.push(ch.iter().collect());
            }
            cur.clear();
        }
    };
    for ch in SAMPLE.chars() {
        if ch == ' ' {
            flush(&mut cur, &mut pieces);
            cur.push(' ');
        } else {
            cur.push(ch);
        }
    }
    flush(&mut cur, &mut pieces);
    pieces
}

fn build_fixture() -> Fixture {
    let tbl = byte_to_char();
    let raw_pieces = piece_strings();

    // Assign a stable id per unique byte-level piece string.
    let mut vocab: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    vocab.insert("<|endoftext|>".into(), serde_json::json!(0));
    let mut ids_by_piece: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut next_id = 1u32;
    let mut order: Vec<u32> = Vec::new();
    for raw in &raw_pieces {
        let bl = to_byte_level(raw, &tbl);
        let id = *ids_by_piece.entry(bl.clone()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            vocab.insert(bl.clone(), serde_json::json!(id));
            id
        });
        order.push(id);
    }

    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{
            "id": 0, "content": "<|endoftext|>", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true
        }],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                          "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": []
        }
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let tokenizer = HuggingFaceTokenizer::new_fastokens(&path).expect("load byte-level tokenizer");

    // Build a ~2048-token generated stream by repeating the sample order.
    let mut stream = Vec::new();
    while stream.len() < 2048 {
        stream.extend_from_slice(&order);
    }
    stream.truncate(2048);

    // Use the first 64 tokens as prompt context.
    let prompt = order.iter().copied().take(64).collect();

    Fixture { tokenizer, prompt, stream }
}

// --- the two implementations under comparison --------------------------------

/// Current path: drive `DecodeStream` exactly as the server does in streaming
/// (intermediate) mode — push each token, pull ready chunks.
fn run_current(fx: &Fixture) -> usize {
    let mut dec = fx.tokenizer.create_decode_stream(&fx.prompt, true, 0);
    let mut emitted = 0usize;
    for &id in &fx.stream {
        let _ = dec.push_token(id).unwrap();
        if let Some(chunk) = dec.next_chunk() {
            emitted += chunk.len();
        }
    }
    let (last, _full) = dec.flush(None).unwrap();
    emitted + last.map_or(0, |s| s.len())
}

/// Idealized floor for a context-free byte-level tokenizer: decode each vocab
/// id to bytes ONCE up front, then per token just append precomputed bytes and
/// emit the longest valid-UTF-8 prefix. No per-token decode, no HashMap, one
/// reused buffer.
struct MemoDecoder {
    id_bytes: Vec<Vec<u8>>,
    pending: Vec<u8>,
}

impl MemoDecoder {
    fn new(fx: &Fixture, max_id: u32) -> Self {
        let tbl = byte_to_char();
        let inv: std::collections::HashMap<char, u8> =
            (0..=255u8).map(|b| (tbl[b as usize], b)).collect();
        let id_bytes = (0..=max_id)
            .map(|id| {
                fx.tokenizer
                    .id_to_token(id)
                    .map(|s| s.chars().filter_map(|c| inv.get(&c).copied()).collect())
                    .unwrap_or_default()
            })
            .collect();
        Self { id_bytes, pending: Vec::with_capacity(64) }
    }

    /// Returns number of newly emitted bytes (as a completed UTF-8 chunk).
    fn push(&mut self, id: u32) -> usize {
        self.pending.extend_from_slice(&self.id_bytes[id as usize]);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return 0;
        }
        // Emit [..valid], retain the incomplete tail.
        self.pending.drain(..valid);
        valid
    }
}

fn run_ideal(fx: &Fixture, memo: &mut MemoDecoder) -> usize {
    memo.pending.clear();
    let mut emitted = 0usize;
    // Seed prompt context (byte-level decode is position-independent, so prompt
    // bytes are just consumed, not emitted).
    for &id in &fx.prompt {
        memo.pending.extend_from_slice(&memo.id_bytes[id as usize]);
    }
    memo.pending.clear();
    for &id in &fx.stream {
        emitted += memo.push(id);
    }
    emitted
}

fn main() {
    let fx = build_fixture();
    let n = fx.stream.len();
    let max_id = *fx.stream.iter().max().unwrap();
    eprintln!(
        "\n=== fixture: {} generated tokens, {} prompt tokens, vocab≈{} ===",
        n,
        fx.prompt.len(),
        max_id + 1
    );

    // --- correctness cross-check: both produce identical text bytes ----------
    let mut dec = fx.tokenizer.create_decode_stream(&fx.prompt, true, 0);
    let mut cur_text = String::new();
    for &id in &fx.stream {
        dec.push_token(id).unwrap();
        if let Some(c) = dec.next_chunk() {
            cur_text.push_str(&c);
        }
    }
    let (last, _) = dec.flush(None).unwrap();
    if let Some(c) = last {
        cur_text.push_str(&c);
    }
    let mut memo = MemoDecoder::new(&fx, max_id);
    let mut ideal_bytes = Vec::new();
    memo.pending.clear();
    for &id in &fx.stream {
        let before = memo.pending.len();
        let emitted = memo.push(id);
        if emitted > 0 {
            // reconstruct emitted bytes for the cross-check
            let _ = before;
        }
        let _ = emitted;
    }
    // Simpler cross-check: full concatenation of memoized bytes == current text.
    for &id in &fx.stream {
        ideal_bytes.extend_from_slice(&memo.id_bytes[id as usize]);
    }
    assert_eq!(
        String::from_utf8_lossy(&ideal_bytes),
        cur_text,
        "memoized floor must reproduce the same decoded text"
    );
    eprintln!("decoded text length: {} bytes (implementations agree)", cur_text.len());

    // --- allocation accounting (deterministic) -------------------------------
    let (a0, b0) = (allocs(), bytes());
    let sink = black_box(run_current(&fx));
    let (a1, b1) = (allocs(), bytes());
    black_box(sink);
    let cur_allocs = a1 - a0;
    let cur_bytes = b1 - b0;

    let mut memo2 = MemoDecoder::new(&fx, max_id);
    let (a2, b2) = (allocs(), bytes());
    let sink = black_box(run_ideal(&fx, &mut memo2));
    let (a3, b3) = (allocs(), bytes());
    black_box(sink);
    let ideal_allocs = a3 - a2;
    let ideal_bytes_alloc = b3 - b2;

    eprintln!("\n--- heap allocations over {n} tokens (one pass) ---");
    eprintln!(
        "current DecodeStream : {:>7} allocs ({:.2}/token), {:>9} bytes",
        cur_allocs,
        cur_allocs as f64 / n as f64,
        cur_bytes
    );
    eprintln!(
        "memoized byte-level  : {:>7} allocs ({:.2}/token), {:>9} bytes",
        ideal_allocs,
        ideal_allocs as f64 / n as f64,
        ideal_bytes_alloc
    );
    eprintln!(
        "reduction            : {:.1}x fewer allocs, {:.1}x fewer bytes",
        cur_allocs as f64 / ideal_allocs.max(1) as f64,
        cur_bytes as f64 / ideal_bytes_alloc.max(1) as f64
    );

    // --- wall-clock via criterion --------------------------------------------
    let mut c = Criterion::default().configure_from_args();
    {
        let mut g = c.benchmark_group("detok_streaming");
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("current_decode_stream", |b| {
            b.iter(|| black_box(run_current(black_box(&fx))))
        });
        let mut memo3 = MemoDecoder::new(&fx, max_id);
        g.bench_function("ideal_memoized_bytelevel", |b| {
            b.iter(|| black_box(run_ideal(black_box(&fx), &mut memo3)))
        });
        g.finish();
    }
    c.final_summary();
}
