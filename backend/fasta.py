from __future__ import annotations

import re
from dataclasses import dataclass


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
INPUT_ALPHABET = re.compile(r"^[A-Za-z*.-]+$")


class FastaError(ValueError):
    """Raised when an uploaded sequence payload is not valid FASTA."""


@dataclass(frozen=True)
class ProteinRecord:
    identifier: str
    sequence: str
    original_length: int
    removed_residues: int


def _clean_sequence(identifier: str, sequence: str) -> ProteinRecord:
    raw = sequence.upper().replace(" ", "").replace("\t", "")
    if not raw:
        raise FastaError(f"序列 {identifier} 为空")
    if not INPUT_ALPHABET.fullmatch(raw):
        invalid = sorted({char for char in raw if not (char.isalpha() or char in "*.-")})
        details = ", ".join(repr(char) for char in invalid[:8])
        raise FastaError(f"序列 {identifier} 含有不支持的字符: {details}")
    cleaned = "".join(char for char in raw if char in STANDARD_AMINO_ACIDS)
    if not cleaned:
        raise FastaError(f"序列 {identifier} 清理非标准残基后为空")
    return ProteinRecord(
        identifier=identifier,
        sequence=cleaned,
        original_length=len(raw),
        removed_residues=len(raw) - len(cleaned),
    )


def parse_fasta(payload: str, *, max_records: int = 50) -> list[ProteinRecord]:
    """Parse FASTA or a single unheaded amino-acid sequence.

    The published preprocessing retained only the 20 standard amino acids.
    Ambiguous residues and alignment punctuation are therefore removed and
    reported to the caller instead of being silently encoded as a new token.
    """

    text = payload.strip()
    if not text:
        raise FastaError("请输入至少一条蛋白质序列")

    if not text.startswith(">"):
        joined = "".join(line.strip() for line in text.splitlines() if line.strip())
        return [_clean_sequence("sequence_1", joined)]

    raw_records: list[tuple[str, list[str]]] = []
    identifier: str | None = None
    chunks: list[str] = []
    seen: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                raw_records.append((identifier, chunks))
            identifier = line[1:].strip().split(maxsplit=1)[0] if line[1:].strip() else ""
            chunks = []
            if not identifier:
                raise FastaError(f"第 {line_number} 行的 FASTA 标题缺少 ID")
            if identifier in seen:
                raise FastaError(f"FASTA ID 重复: {identifier}")
            seen.add(identifier)
        elif identifier is None:
            raise FastaError(f"第 {line_number} 行的序列出现在首个 FASTA 标题之前")
        else:
            chunks.append(line)

    if identifier is not None:
        raw_records.append((identifier, chunks))
    if not raw_records:
        raise FastaError("没有找到 FASTA 记录")
    if len(raw_records) > max_records:
        raise FastaError(f"单次最多提交 {max_records} 条序列，当前为 {len(raw_records)} 条")

    return [_clean_sequence(name, "".join(parts)) for name, parts in raw_records]
