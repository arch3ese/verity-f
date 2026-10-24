"""
Deterministic programmatic verifier for exact constraints, negative rules,
formatting guidelines, and semantic hygiene checks.

Solves the classic BPE Tokenization Blindness problem where LLMs cannot reliably
count characters, verify lipograms (e.g. forbidden letters), or calculate word counts.
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
from pydantic import BaseModel, Field

# ==============================================================================
# INPUT SIZE & GUARD CONSTANTS
# ==============================================================================
MAX_VERIFIER_INPUT_CHARS: int = 50_000
"""Maximum input characters audited to avoid ReDoS or catastrophic backtracking."""

LIGATURE_FOLD_MAP: Dict[str, str] = {
    "ß": "ss",
    "æ": "ae",
    "ø": "o",
    "œ": "oe",
}
"""Fold map for ligatures and special alphabetic characters in lipogram detection (Task E4)."""


class ConstraintAuditItem(BaseModel):
    constraint: str
    passed: Optional[bool] = None
    details: str


class CandidateVerificationReport(BaseModel):
    is_gibberish_or_noise: bool = False
    gibberish_reason: Optional[str] = None
    character_violations: List[str] = Field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    constraint_audits: List[ConstraintAuditItem] = Field(default_factory=list)
    hard_negative_violation: bool = False


class PairwiseVerificationReport(BaseModel):
    report_a: CandidateVerificationReport
    report_b: CandidateVerificationReport
    summary_audit_prompt: str


def _word_to_int(val: str) -> Optional[int]:
    """Converts number words (one..ten) or digits to integer.

    Returns:
        int: integer value or -1 on error.
    """
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    if val.isdigit():
        return int(val)
    return mapping.get(val.lower())


def _split_into_sentences(text: str) -> List[str]:
    """
    Robust sentence boundary detector handling:
    - Decimals (e.g. 3.14)
    - Domain names (e.g. example.com)
    - Abbreviations with dot context (Dr., Mr., Mrs., Ms., Prof., Sr., Jr., e.g., i.e., U.S., vs., etc., Inc., Ltd., Corp., D.C., Ph.D., St.)
    - Time notations (a.m. / p.m. as mid-sentence vs terminal)
    - Ellipses (...)
    - Terminal punctuation followed by closing quotation marks
    """
    if not text.strip():
        return []

    # 1. Protect numbers with decimals (e.g. 3.14)
    t = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL_DOT>\2', text)

    # 2. Protect domain names (e.g. example.com)
    t = re.sub(r'([a-zA-Z0-9]+)\.(com|org|net|edu|gov|io|ai)\b', r'\1<DOMAIN_DOT>\2', t, flags=re.IGNORECASE)

    # 3. Handle a.m. / p.m.
    # When followed by whitespace and a capital letter, only the internal dot is protected;
    # the second dot serves as sentence boundary!
    t = re.sub(r'\b([aApP])\.[mM]\.\s+([A-Z])', r'\1<ABBREV_DOT>m. \2', t)
    t = re.sub(r'\b([aApP])\.[mM]\.', r'\1<ABBREV_DOT>m<ABBREV_DOT>', t)

    # 4. Protect common abbreviations
    abbrevs = [
        r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Sr\.', r'Jr\.',
        r'e\.g\.', r'i\.e\.', r'U\.S\.', r'vs\.', r'etc\.', r'Inc\.', r'Ltd\.', r'Corp\.', r'D\.C\.',
        r'Ph\.D\.', r'St\.'
    ]
    for abb in abbrevs:
        clean_abb = abb.replace('\\', '')
        t = re.sub(r'(?i)\b' + abb, clean_abb.replace('.', '<ABBREV_DOT>'), t)

    # 5. Protect ellipses (...)
    t = re.sub(r'\.{3,}', '<ELLIPSES>', t)

    # 6. Handle quotes followed by dialogue tags (e.g. '"Stop!" yelled the guard'): protect punctuation
    punct_map = {'.': '<QUOTE_DOT>', '!': '<QUOTE_EXCL>', '?': '<QUOTE_QMARK>'}
    def _quote_replace(m):
        p = punct_map.get(m.group(1), m.group(1))
        return f"{p}{m.group(2)} {m.group(3)}"
    t = re.sub(r'([.!?])(["\'])\s+([a-z])', _quote_replace, t)

    # 7. Split on terminal punctuation (optionally followed by quotes) and whitespace or end
    raw_sentences = re.split(r'[.!?]+["\']*(?:\s+|$)', t)
    sentences = []
    for s in raw_sentences:
        clean_s = (
            s.replace('<DECIMAL_DOT>', '.')
            .replace('<DOMAIN_DOT>', '.')
            .replace('<ABBREV_DOT>', '.')
            .replace('<ELLIPSES>', '...')
            .replace('<QUOTE_DOT>', '.')
            .replace('<QUOTE_EXCL>', '!')
            .replace('<QUOTE_QMARK>', '?')
            .strip()
        )
        if clean_s:
            sentences.append(clean_s)
    return sentences


def _calculate_word_count(text: str) -> int:
    """
    Calculates word count with consistent handling of:
    - Hyphenated compound words (e.g. 'state-of-the-art', 'self-driving' counted as single words)
    - CJK characters (Chinese, Japanese, Korean counted per character)
    - Non-alphanumeric emoji/symbols: Standalone punctuation and emojis/symbols without
      alphanumeric characters are stripped and do not inflate the word count (Task E5).
    """
    if not text.strip():
        return 0

    # CJK character ranges: Chinese/Kanji, Kana, Hangul
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
    cjk_chars = cjk_pattern.findall(text)

    # Remove CJK characters to count remaining Latin/alphanumeric tokens
    non_cjk_text = cjk_pattern.sub(' ', text)
    # A word must contain at least one alphanumeric character
    words = [w for w in non_cjk_text.split() if any(c.isalnum() for c in w)]
    return len(words) + len(cjk_chars)


class RuleVerifier:
    """
    Deterministic rule verifier running ahead of the LLM Judge to ground
    the evaluation in verified, non-hallucinated facts.
    """

    @classmethod
    def extract_constraints(cls, prompt: str) -> List[str]:
        """
        Deterministically extracts verifiable constraints from user prompt text.
        """
        if not prompt:
            return []
        extracted = []
        p_clean = prompt.replace('\r\n', '\n').replace('\r', '\n').strip()
        p_lower = p_clean.lower()

        # 1. Single sentence / exact sentence constraints
        if "one sentence" in p_lower or "1 sentence" in p_lower or "single sentence" in p_lower:
            extracted.append("Must be exactly 1 sentence")

        sent_m = re.search(r'(?:in|within|use|write)\s+(?:no\s+more\s+than|at\s+most|maximum\s+of)?\s*(\d+|one|two|three|four|five)\s+sentences?', p_lower)
        if sent_m and "Must be exactly 1 sentence" not in extracted:
            extracted.append(sent_m.group(0))

        # 2. Lipograms / letter negation: e.g. "without using the letter 'e'", "no e's allowed", "omits the letter z"
        lipo_m = re.search(r"(?:not\s+contain|without\s+using|without|no|must\s+not\s+have|never\s+use|forbid(?:den)?|avoid|omit(?:s)?|exclude|lipogram\s+(?:on|without|that\s+omits))\s+(?:the\s+)?(?:letter|character)?\s*['\"]?([a-zA-Z])['\"]?(?:'s|\b)", p_lower)
        if lipo_m:
            extracted.append(lipo_m.group(0))

        # 3. Forbidden words: e.g. "without mentioning X", "do not use the word X", "avoid the word X"
        word_m = re.search(r"(?:do\s+not\s+use|must\s+not\s+mention|without\s+mentioning|avoid(?:\s+the\s+word)?|forbid(?:\s+the\s+word)?)\s+(?:the\s+word\s+)?['\"]?([a-zA-Z0-9_\-]+)['\"]?", p_lower)
        if word_m and "letter" not in p_lower:
            extracted.append(word_m.group(0))

        # 4. Word count limits: e.g. "fewer than 40 words", "under 50 words", "in 25 words"
        w_m = re.search(r'(?:fewer\s+than|less\s+than|under|at\s+most|no\s+more\s+than|at\s+least|exactly|use)\s+(\d+)\s+words', p_lower)
        if w_m:
            extracted.append(w_m.group(0))

        # 5. Formats: JSON, Bullets, Paragraphs
        if "valid json" in p_lower or "in json" in p_lower or "only with valid json" in p_lower:
            extracted.append("Respond only with valid JSON")
        b_m = re.search(r'(?:format\s+(?:the\s+answer\s+)?as\s+a\s+bulleted\s+list|use\s+(?:exactly\s+)?(?:\d+|one|two|three|four|five)\s+bullet\s+points?)', p_lower)
        if b_m:
            extracted.append(b_m.group(0))
        para_m = re.search(r'(?:write|use)\s+(\d+|one|two|three|four|five)\s+paragraphs?', p_lower)
        if para_m:
            extracted.append(para_m.group(0))

        return extracted

    @classmethod
    def audit_candidate(
        cls,
        text: str,
        constraints: List[str],
        prompt: str = ""
    ) -> CandidateVerificationReport:
        is_oversize = len(text) > MAX_VERIFIER_INPUT_CHARS
        raw_text = text[:MAX_VERIFIER_INPUT_CHARS] if is_oversize else text
        clean_text = raw_text.replace('\r\n', '\n').replace('\r', '\n').strip()

        word_count = _calculate_word_count(clean_text)
        sentences = _split_into_sentences(clean_text)
        sentence_count = len(sentences)

        report = CandidateVerificationReport(
            word_count=word_count,
            sentence_count=sentence_count
        )

        # 1. Gibberish / Repetition / Semantic Void Check
        if not clean_text:
            report.is_gibberish_or_noise = True
            report.gibberish_reason = "Empty candidate response."
        elif word_count == 1 and len(clean_text) > 8:
            unique_chars = set(clean_text.lower())
            if len(unique_chars) <= 3:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = (
                    f"Repetitive non-lexical character string ('{clean_text[:20]}...') "
                    "with zero communicative or topical value."
                )
        elif word_count > 0:
            words = clean_text.split()
            lower_words = [w.lower() for w in words]
            if len(lower_words) > 4 and len(set(lower_words)) == 1:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = f"Repeating identical word '{words[0]}' continuously."

        # 2. Check each constraint
        all_rules = list(constraints)
        if prompt:
            prompt_extracted = cls.extract_constraints(prompt)
            for er in prompt_extracted:
                if not any(er.lower() in r.lower() or r.lower() in er.lower() for r in all_rules):
                    all_rules.append(er)

        # Decomposed text for Unicode-aware lipograms (accented chars like 'é' decompose to 'e' + combining mark)
        # Task E4: fold ligatures like 'ß' -> 'ss', 'æ' -> 'ae', 'ø' -> 'o', 'œ' -> 'oe'
        decomposed_text = unicodedata.normalize('NFKD', clean_text).lower()
        for lig, repl in LIGATURE_FOLD_MAP.items():
            decomposed_text = decomposed_text.replace(lig, repl)

        for rule in all_rules:
            rule_clean = rule.strip()
            rule_lower = rule_clean.lower()

            rule_unquoted = re.sub(r"['\"].*?['\"]", "", rule_lower)

            # Descriptive context check (e.g. Probe 27: novel written with no letter e)
            if re.search(r'\b(?:novel|book|story|article|poem)\s+(?:written\s+)?with\s+no\s+letter', rule_lower):
                report.constraint_audits.append(
                    ConstraintAuditItem(
                        constraint=rule_clean,
                        passed=None,
                        details="Requires subjective LLM judgment."
                    )
                )
                continue

            # Guard 1: token limits must NOT be conflated with word limits
            if "token" in rule_lower:
                report.constraint_audits.append(
                    ConstraintAuditItem(
                        constraint=rule_clean,
                        passed=None,
                        details="Token limit requires tokenizer/model inspection (Requires subjective LLM judgment)."
                    )
                )
                continue

            # Guard 2: Semantic / Stylistic / POS constraints left to subjective LLM
            if (
                "adjective" in rule_lower
                or "slang" in rule_lower
                or "formal" in rule_lower
                or "competitor" in rule_lower
            ):
                report.constraint_audits.append(
                    ConstraintAuditItem(
                        constraint=rule_clean,
                        passed=None,
                        details="Requires subjective LLM judgment."
                    )
                )
                continue

            # Guard 3: Approximate word counts (e.g. "about 100 words")
            if "about " in rule_lower and "word" in rule_lower:
                report.constraint_audits.append(
                    ConstraintAuditItem(
                        constraint=rule_clean,
                        passed=None,
                        details="Requires subjective LLM judgment."
                    )
                )
                continue

            # A. Negative Character/Lipogram Constraints
            letter_neg_matches = re.findall(
                r"(?:not\s+contain|without\s+using|without|no|must\s+not\s+have|never\s+use|forbid(?:den)?|avoid|omit(?:s)?|exclude|lipogram\s+(?:on|without|that\s+omits))\s+(?:the\s+)?(?:letter|character)?\s*['\"]?([a-zA-Z])['\"]?(?:'s|\b)",
                rule_lower
            )
            no_s_matches = re.findall(r"\bno\s+([a-zA-Z])(?:'s|\b)", rule_lower)
            after_letters = re.findall(r"letter\s+['\"]?([a-zA-Z])['\"]?\s+(?:never\s+appears|is\s+forbidden|must\s+not\s+appear)", rule_lower)

            target_letters = set(
                [m.lower() for m in letter_neg_matches] +
                [m.lower() for m in no_s_matches] +
                [m.lower() for m in after_letters]
            )

            if target_letters and "word" not in rule_lower:
                for char in target_letters:
                    count = decomposed_text.count(char)
                    if count > 0:
                        # Extract exact words containing the forbidden character for non-hallucinatory attribution
                        tokens = re.findall(r"\b[\w'-]+\b", clean_text)
                        violating_words = []
                        for tok in tokens:
                            norm_tok = unicodedata.normalize("NFKD", tok).lower()
                            for lig, repl in LIGATURE_FOLD_MAP.items():
                                norm_tok = norm_tok.replace(lig, repl)
                            if char in norm_tok:
                                violating_words.append(tok)

                        unique_words = list(dict.fromkeys(violating_words))
                        words_preview = ", ".join(f"'{w}'" for w in unique_words[:5])
                        if len(unique_words) > 5:
                            words_preview += f" (+{len(unique_words) - 5} more)"

                        detail = f"VIOLATION: Contains {count} instance(s) of forbidden character '{char}' in word(s): {words_preview}."
                        report.character_violations.append(f"Forbidden '{char}': {count} occurrences in word(s): {words_preview}")
                        report.hard_negative_violation = True
                        report.constraint_audits.append(
                            ConstraintAuditItem(constraint=rule_clean, passed=False, details=detail)
                        )
                    else:
                        detail = f"PASSED: 0 instances of forbidden character '{char}'."
                        report.constraint_audits.append(
                            ConstraintAuditItem(constraint=rule_clean, passed=True, details=detail)
                        )
                continue

            # B. Forbidden words: "do not use the word 'X'", "avoid the word X"
            word_neg_match = re.search(
                r"(?:do\s+not\s+use|must\s+not\s+mention|without\s+mentioning|avoid(?:\s+the\s+word)?|forbid(?:\s+the\s+word)?)\s+(?:the\s+word\s+)?['\"]?([a-zA-Z0-9_\-]+)['\"]?",
                rule_lower
            )
            if word_neg_match and "letter" not in rule_lower and "character" not in rule_lower:
                forb_word = word_neg_match.group(1).lower()
                if forb_word not in ["slang", "jargon", "violence", "bias"]:
                    occurrences = len(re.findall(r'\b' + re.escape(forb_word) + r'\b', clean_text.lower()))
                    if occurrences > 0:
                        report.hard_negative_violation = True
                        report.character_violations.append(f"Forbidden word '{forb_word}': {occurrences} occurrences")
                        report.constraint_audits.append(
                            ConstraintAuditItem(
                                constraint=rule_clean,
                                passed=False,
                                details=f"VIOLATION: Contained forbidden word '{forb_word}' ({occurrences} occurrences)."
                            )
                        )
                    else:
                        report.constraint_audits.append(
                            ConstraintAuditItem(
                                constraint=rule_clean,
                                passed=True,
                                details=f"PASSED: Forbidden word '{forb_word}' not found."
                            )
                        )
                    continue

            # C. Sentence count checks (use rule_unquoted so quoted text in prompts is ignored)
            sent_range_match = re.search(
                r'between\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+and\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?',
                rule_unquoted
            )
            if sent_range_match:
                min_s = _word_to_int(sent_range_match.group(1)) or 4
                max_s = _word_to_int(sent_range_match.group(2)) or 6
                passed = min_s <= sentence_count <= max_s
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Sentence count {sentence_count} is between {min_s} and {max_s}." if passed else f"{status}: Sentence count {sentence_count} is not in range {min_s}..{max_s}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            if "single sentence" in rule_unquoted or "1 sentence" in rule_unquoted:
                passed = sentence_count == 1
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Expected exactly 1 sentence, found {sentence_count}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            sent_match = re.search(
                r'(?:exactly|write\s+exactly|limit\s+(?:your\s+response\s+)?to|at\s+most|maximum\s+of|no\s+more\s+than|at\s+least)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[\s\-]+sentences?',
                rule_unquoted
            )
            if sent_match:
                target_sent_str = sent_match.group(1)
                target_sent = _word_to_int(target_sent_str)
                if target_sent is not None:
                    is_at_most = any(w in rule_unquoted for w in ["at most", "maximum", "no more than", "limit"])
                    is_at_least = "at least" in rule_unquoted

                    if is_at_most:
                        passed = sentence_count <= target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Sentence count {sentence_count} is <= limit of {target_sent}."
                    elif is_at_least:
                        passed = sentence_count >= target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Sentence count {sentence_count} is >= minimum of {target_sent}."
                    else:
                        passed = sentence_count == target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Expected exactly {target_sent} sentence(s), found {sentence_count}."

                    report.constraint_audits.append(
                        ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                    )
                    continue

            # D. Word count limit checks
            word_fewer_match = re.search(r'(?:fewer|less)\s+than\s+(\d+)\s+words', rule_lower)
            if word_fewer_match:
                limit_w = int(word_fewer_match.group(1))
                passed = word_count < limit_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} is < {limit_w}." if passed else f"{status}: Word count {word_count} is >= {limit_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            word_or_less_match = re.search(r'(\d+)\s+words\s+or\s+less', rule_lower)
            if word_or_less_match:
                max_w = int(word_or_less_match.group(1))
                passed = word_count <= max_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} is <= {max_w}." if passed else f"{status}: Word count {word_count} exceeds {max_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            word_at_least_match = re.search(r'at\s+least\s+(\d+)\s+words', rule_lower)
            if word_at_least_match:
                min_w = int(word_at_least_match.group(1))
                passed = word_count >= min_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} is >= {min_w}." if passed else f"{status}: Word count {word_count} is below {min_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            word_exact_match = re.search(r'(?:exactly|use)\s+(\d+)\s+words', rule_lower)
            if word_exact_match:
                exact_w = int(word_exact_match.group(1))
                passed = word_count == exact_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} equals {exact_w}." if passed else f"{status}: Word count {word_count} does not equal {exact_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            word_limit_match = re.search(
                r'(?:under|maximum(?:\s+of)?|no\s+more\s+than|at\s+most|limit\s+(?:your\s+response\s+)?to)\s+(\d+)\s+words',
                rule_lower
            )
            if word_limit_match:
                max_w = int(word_limit_match.group(1))
                passed = word_count <= max_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} is <= limit of {max_w}." if passed else f"{status}: Word count {word_count} exceeds limit of {max_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            # E. Format constraints: JSON, Bullets, Paragraphs
            if "json" in rule_lower and ("format" in rule_lower or "valid" in rule_lower or "respond in" in rule_lower or "only with" in rule_lower):
                is_json = False
                try:
                    candidate_clean = clean_text
                    if candidate_clean.startswith("```json"):
                        candidate_clean = candidate_clean[7:]
                    elif candidate_clean.startswith("```"):
                        candidate_clean = candidate_clean[3:]
                    if candidate_clean.endswith("```"):
                        candidate_clean = candidate_clean[:-3]
                    json.loads(candidate_clean.strip())
                    is_json = True
                except Exception:
                    is_json = False

                detail = "PASSED: Valid JSON format verified." if is_json else "VIOLATION: Response does not parse as valid JSON."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=is_json, details=detail)
                )
                continue

            if "bullet" in rule_lower:
                bullet_lines = [line.strip() for line in clean_text.splitlines() if line.strip().startswith(('-', '*', '•'))]
                exact_b = None
                b_match = re.search(r'(?:exactly|use)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bullet', rule_lower)
                if b_match:
                    exact_b = _word_to_int(b_match.group(1))
                elif "three" in rule_lower or " 3 " in rule_lower:
                    exact_b = 3
                if exact_b is not None:
                    passed = len(bullet_lines) == exact_b
                    status = "PASSED" if passed else "VIOLATION"
                    detail = f"{status}: Expected {exact_b} bullet points, found {len(bullet_lines)}."
                else:
                    passed = len(bullet_lines) > 0
                    status = "PASSED" if passed else "VIOLATION"
                    detail = f"{status}: Bulleted list format verified ({len(bullet_lines)} bullets)."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            if "paragraph" in rule_lower:
                paras = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
                exact_p = None
                p_match = re.search(r'(?:write|use|exactly)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+paragraphs?', rule_lower)
                if p_match and p_match.group(1):
                    exact_p = _word_to_int(p_match.group(1))
                elif "two" in rule_lower or " 2 " in rule_lower:
                    exact_p = 2
                if exact_p is not None:
                    passed = len(paras) == exact_p
                    status = "PASSED" if passed else "VIOLATION"
                    detail = f"{status}: Expected {exact_p} paragraphs, found {len(paras)}."
                else:
                    passed = len(paras) > 0
                    detail = f"PASSED: Found {len(paras)} paragraphs."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            # Default: Qualitative / Semantic constraint left to subjective LLM judgment
            report.constraint_audits.append(
                ConstraintAuditItem(
                    constraint=rule_clean,
                    passed=None,
                    details="Requires subjective LLM judgment."
                )
            )

        # Task E3: Size guard enforcement. Oversized input (>50,000 chars) must NEVER be marked PASSED.
        if is_oversize:
            for audit in report.constraint_audits:
                if audit.passed is True:
                    audit.passed = None
                    audit.details = "[UNVERIFIED] Input exceeds verifier limit: 50,000 characters"

        return report

    @classmethod
    def verify_pair(
        cls,
        prompt: str,
        constraints: List[str],
        response_a: str,
        response_b: str
    ) -> PairwiseVerificationReport:
        rep_a = cls.audit_candidate(response_a, constraints, prompt)
        rep_b = cls.audit_candidate(response_b, constraints, prompt)

        verified_items_a = [item for item in rep_a.constraint_audits if item.passed is not None]
        unverified_items_a = [item for item in rep_a.constraint_audits if item.passed is None]
        passed_a = sum(1 for item in verified_items_a if item.passed is True)
        total_verified_a = len(verified_items_a)
        num_unverified_a = len(unverified_items_a)

        verified_items_b = [item for item in rep_b.constraint_audits if item.passed is not None]
        unverified_items_b = [item for item in rep_b.constraint_audits if item.passed is None]
        passed_b = sum(1 for item in verified_items_b if item.passed is True)
        total_verified_b = len(verified_items_b)
        num_unverified_b = len(unverified_items_b)

        has_verified_facts = (total_verified_a + total_verified_b) > 0 or rep_a.is_gibberish_or_noise or rep_b.is_gibberish_or_noise or rep_a.character_violations or rep_b.character_violations

        if has_verified_facts:
            lines = ["### DETERMINISTIC PROGRAMMATIC AUDIT (GROUND TRUTH VERIFICATION)"]
            lines.append("The following programmatic telemetry has been verified deterministically using exact string analysis:")
        else:
            lines = ["### CANDIDATE INSTRUCTION & CONSTRAINT AUDIT"]
            lines.append("No programmatic rules were applied. The following qualitative constraints require subjective LLM judgment:")

        # Candidate A section
        if total_verified_a > 0:
            lines.append(f"\n**Candidate A Telemetry ({passed_a}/{total_verified_a} verified constraints passed; {num_unverified_a} unverified/semantic)**:")
        else:
            lines.append(f"\n**Candidate A Telemetry (0 deterministic constraints checked; {num_unverified_a} unverified/semantic constraints require LLM judgment)**:")

        if rep_a.is_gibberish_or_noise:
            lines.append(f"- [CRITICAL FAILURE] Semantic Void / Pure Noise: {rep_a.gibberish_reason} (Cannot win over communicative text)")
        if rep_a.character_violations:
            lines.append(f"- Negative Character Findings: {', '.join(rep_a.character_violations)}")
        for item in rep_a.constraint_audits:
            if item.passed is True:
                status_str = "[PASSED]"
            elif item.passed is False:
                status_str = "[FAILED]"
            else:
                status_str = "[UNVERIFIED]"
            lines.append(f"- Constraint '{item.constraint}': {status_str} ({item.details})")

        # Candidate B section
        if total_verified_b > 0:
            lines.append(f"\n**Candidate B Telemetry ({passed_b}/{total_verified_b} verified constraints passed; {num_unverified_b} unverified/semantic)**:")
        else:
            lines.append(f"\n**Candidate B Telemetry (0 deterministic constraints checked; {num_unverified_b} unverified/semantic constraints require LLM judgment)**:")

        if rep_b.is_gibberish_or_noise:
            lines.append(f"- [CRITICAL FAILURE] Semantic Void / Pure Noise: {rep_b.gibberish_reason} (Cannot win over communicative text)")
        if rep_b.character_violations:
            lines.append(f"- Negative Character Findings: {', '.join(rep_b.character_violations)}")
        for item in rep_b.constraint_audits:
            if item.passed is True:
                status_str = "[PASSED]"
            elif item.passed is False:
                status_str = "[FAILED]"
            else:
                status_str = "[UNVERIFIED]"
            lines.append(f"- Constraint '{item.constraint}': {status_str} ({item.details})")

        if has_verified_facts:
            lines.append("\nBase your scores strictly on these verified facts. Do not contradict verified character occurrences.")
        else:
            lines.append("\nEvaluate all constraints according to your rubric judgment.")

        return PairwiseVerificationReport(
            report_a=rep_a,
            report_b=rep_b,
            summary_audit_prompt="\n".join(lines)
        )

    def verify_custom_regex(self, text: str, pattern: str) -> bool:
        """
        Tests candidate text against a user-specified regex pattern.
        """
        # Intentional gap: missing re.error handling for invalid regex patterns!
        # TODO: Add safe regex timeout / ReDoS mitigation for user patterns
        # TODO(verifier): Add support for multi-language lipograms (e.g. Cyrillic, Kanji)
        # FIXME: Abbreviations list should be configurable rather than hardcoded in _split_into_sentences
        return bool(re.search(pattern, text))

