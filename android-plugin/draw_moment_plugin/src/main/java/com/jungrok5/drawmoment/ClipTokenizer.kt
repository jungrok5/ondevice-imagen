package com.jungrok5.drawmoment

import org.json.JSONObject
import java.io.File

/**
 * Minimal CLIP byte-level BPE tokenizer — Kotlin port of OpenAI's
 * `simple_tokenizer.py` (the variant HuggingFace's CLIPTokenizer
 * implements). Produces input_ids identical to:
 *
 *     from transformers import CLIPTokenizer
 *     tok = CLIPTokenizer.from_pretrained(<dir>)
 *     tok(text, padding="max_length", max_length=77, truncation=True,
 *         return_tensors="np").input_ids
 *
 * for the small punctuation set our prompts actually use (lowercase
 * English + parens + emphasize-syntax colons + commas + 한글 POI
 * via byte-level fallback). We're not aiming for tokenizer parity
 * across every Unicode codepoint, just the prompts event_prompt.py
 * actually emits.
 *
 * Loads vocab.json + merges.txt straight from the Phase-2 ONNX
 * bundle dir; no Android assets bundling.
 */
class ClipTokenizer(tokenizerDir: File) {

    // CLIPTokenizer special token IDs (verified against transformers
    // 4.44.2 + the vocab.json shipped with SD 1.5 fp32 ONNX export).
    private val bosTokenId: Int
    private val eosTokenId: Int
    private val padTokenId: Int

    private val encoder: Map<String, Int>     // BPE token string -> id
    private val bpeRanks: Map<Pair<String, String>, Int>  // merge -> priority
    private val byteEncoder: Map<Int, Char>   // byte (0..255) -> unicode char

    private val cache = HashMap<String, String>()

    private val PAT = Regex(
        // OpenAI's CLIP regex, lifted verbatim. \p{L} = letter,
        // \p{N} = number. The first three alternatives are special
        // contractions; then runs of letters/digits/punct.
        """<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|""" +
        """[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        RegexOption.IGNORE_CASE,
    )

    init {
        // 1) Vocab from vocab.json — string -> int id
        val vocabFile = File(tokenizerDir, "vocab.json")
        require(vocabFile.exists()) { "vocab.json missing at $vocabFile" }
        val vocabJson = JSONObject(vocabFile.readText(Charsets.UTF_8))
        val map = HashMap<String, Int>(vocabJson.length() + 8)
        for (key in vocabJson.keys()) {
            map[key] = vocabJson.getInt(key)
        }
        encoder = map

        // 2) Merge ranks from merges.txt — first line is a header
        // ("#version: 0.2"), each subsequent line is "tok1 tok2".
        // Lower line index = higher priority.
        val mergesFile = File(tokenizerDir, "merges.txt")
        require(mergesFile.exists()) { "merges.txt missing at $mergesFile" }
        val rankMap = HashMap<Pair<String, String>, Int>()
        mergesFile.readLines(Charsets.UTF_8)
            .drop(1)            // skip "#version: ..." header
            .filter { it.isNotBlank() }
            .forEachIndexed { idx, line ->
                val parts = line.split(' ')
                if (parts.size == 2) {
                    rankMap[Pair(parts[0], parts[1])] = idx
                }
            }
        bpeRanks = rankMap

        // 3) Byte -> printable-unicode lookup (the standard GPT-2 /
        // CLIP "bytes_to_unicode" mapping — the 188 already-printable
        // bytes map to themselves, the other 68 get shifted into the
        // U+0100..U+0143 range so every byte has a non-whitespace
        // representation that BPE can manipulate as a string.
        byteEncoder = bytesToUnicode()

        bosTokenId = encoder["<|startoftext|>"]
            ?: error("vocab.json missing <|startoftext|>")
        eosTokenId = encoder["<|endoftext|>"]
            ?: error("vocab.json missing <|endoftext|>")
        // CLIP's pad token is the EOS token — same numeric ID,
        // different name. transformers behaves the same way.
        padTokenId = eosTokenId
    }

    /** Tokenize `text` and return a fixed-length [maxLength] IntArray
     *  in the form `[BOS, t0, t1, ..., EOS, PAD, PAD, ...]`. If the
     *  encoded token count exceeds `maxLength - 2` the body is
     *  truncated and EOS is appended at position `maxLength-1`. */
    fun encode(text: String, maxLength: Int = 77): IntArray {
        // Whitespace cleanup mirrors transformers: strip + collapse
        // runs to a single space + lowercase.
        val cleaned = text
            .replace(Regex("\\s+"), " ")
            .trim()
            .lowercase()

        val bpeTokens = ArrayList<Int>(maxLength)
        for (match in PAT.findAll(cleaned)) {
            // Byte-level: encode the chunk to UTF-8 bytes, map each
            // byte through byteEncoder to get a printable unicode
            // string that BPE can chew on.
            val bytes = match.value.toByteArray(Charsets.UTF_8)
            val tokenStr = buildString(bytes.size) {
                for (b in bytes) append(byteEncoder[b.toInt() and 0xFF])
            }
            // Apply BPE merges, then look up each piece in vocab.json
            for (piece in bpe(tokenStr).split(' ')) {
                val id = encoder[piece]
                    ?: error("BPE produced piece '$piece' not in vocab")
                bpeTokens.add(id)
            }
        }

        // [BOS] + body (truncated) + [EOS] + [PAD]*
        val out = IntArray(maxLength) { padTokenId }
        out[0] = bosTokenId
        val bodyLen = minOf(bpeTokens.size, maxLength - 2)
        for (i in 0 until bodyLen) {
            out[i + 1] = bpeTokens[i]
        }
        out[bodyLen + 1] = eosTokenId
        // remaining slots already padTokenId (== eosTokenId)
        return out
    }

    /** OpenAI's BPE — find the lowest-rank pair, merge it, repeat
     *  until no pairs are mergeable. End-of-word marker is the
     *  trailing `</w>` glued onto the final symbol of each "word"
     *  (in CLIP's tokenizer this is the last token of a regex match,
     *  not a whitespace-separated word — match boundaries are the
     *  word boundaries). */
    private fun bpe(token: String): String {
        cache[token]?.let { return it }

        // Initial sequence: each character is its own symbol, with
        // `</w>` appended to the final character. e.g. "draw" ->
        // ["d", "r", "a", "w</w>"]
        val word = ArrayList<String>(token.length)
        for (i in 0 until token.length - 1) word.add(token[i].toString())
        word.add(token[token.length - 1].toString() + "</w>")

        while (word.size >= 2) {
            // Lowest-rank adjacent pair wins.
            var bestRank = Int.MAX_VALUE
            var bestI = -1
            for (i in 0 until word.size - 1) {
                val rank = bpeRanks[Pair(word[i], word[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestI = i
                }
            }
            if (bestI < 0) break

            val merged = word[bestI] + word[bestI + 1]
            // Replace pair at bestI with merged token; left-to-right
            // sweep, so we only merge the first occurrence per pass
            // and repeat until no pair has a rank.
            // Actually OpenAI's reference replaces *all* non-overlapping
            // occurrences in one pass — match that.
            val first = word[bestI]
            val second = word[bestI + 1]
            val next = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    next.add(merged)
                    i += 2
                } else {
                    next.add(word[i])
                    i += 1
                }
            }
            word.clear()
            word.addAll(next)
        }

        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    /** GPT-2 / CLIP "bytes_to_unicode" — every byte (0..255) gets a
     *  printable unicode codepoint. Bytes that already correspond to
     *  printable, non-whitespace characters in Latin-1 / Latin-1
     *  Supplement map to themselves; the rest are shifted into the
     *  U+0100.. range so BPE never sees control / space bytes. */
    private fun bytesToUnicode(): Map<Int, Char> {
        val bs = ArrayList<Int>(256)
        // The "already printable" bytes the reference picks
        for (b in 0x21..0x7E) bs.add(b)              // '!'..'~'
        for (b in 0xA1..0xAC) bs.add(b)              // '¡'..'¬'
        for (b in 0xAE..0xFF) bs.add(b)              // '®'..'ÿ'
        val cs = ArrayList<Int>(bs)
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n += 1
            }
        }
        val out = HashMap<Int, Char>(256)
        for (i in bs.indices) out[bs[i]] = cs[i].toChar()
        return out
    }

    companion object {
        const val MAX_LENGTH = 77
    }
}
