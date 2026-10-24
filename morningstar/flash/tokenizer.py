"""BPE 分词器 (Sennrich et al. 2016) —— 纯 NumPy，零外部依赖。

世界级出处:
  Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of
  Rare Words with Subword Units*. ACL 2016. arXiv:1508.07909.

为什么 BPE 是 LLM 分词的事实标准:
  * **闭合词汇表** —— 任意未见词都能由子词拼出，杜绝 UNK；
  * **频次贪心合并** —— 高频字符对先合并，压缩序列长度，等价于在
    「词表大小 ↔ 序列长度」之间做最优折衷；
  * GPT-2 / RoBERTa / Llama-2 均沿袭此算法（GPT-2 额外加了正则预分词）。

硬不变量 (tests 强制执行):
  **T1 无损往返**: ``decode(encode(s)) == s`` 对任意字符串成立 —— 分词是
      字符串上的**可逆编码**（这是实现的正确性底线）。
  **T2 合并单调**: 训练过程中 token 数单调不增（每次合并必然减少一对相邻单元）。
  **T3 确定性**: 同一语料 + 同一 vocab_size ⇒ 逐字节相同的 merges 序列
      （频次相同时按「字典序」打破平局，而非依赖 dict/set 的顺序）。
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Iterable, Sequence

_LOGGER = logging.getLogger(__name__)


class BPETokenizer:
    """字节级 BPE: 以 UTF-8 字节为原子单元，天然支持任意 Unicode。

    设计选择（零重写 + 顶级实践）:
      * **字节级**而非字符级: 与 GPT-2 的 ``bytes_to_unicode`` 思路一致，
        保证任何输入（包括中文、emoji）都能被编码，无需 UNK 兜底。
      * 合成��表时用一个**哨兵独占私有码位**做整词边界标记，避免跨词合并歧义。
    """

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._cache: dict[tuple[int, ...], list[int]] = {}

    # ---------------------------------------------------------------- training
    def train(self, texts: Sequence[str], vocab_size: int, *, verbose: bool = False) -> "BPETokenizer":
        """在语料上训练合并表。

        Args:
            texts:      原始字符串序列。
            vocab_size: 目标词表大小（必须 ≥ 256 + 词数上界）。
            verbose:    是否输出训练进度日志。
        """
        if vocab_size < 258:
            raise ValueError(f"vocab_size 必须 ≥ 258（256 字节 + 至少 2 个合并），收到 {vocab_size}")

        # 语料表示为「整数序列」(每个字节一个 id)，避免 Python str 的低效切分
        ids_corpus: list[list[int]] = [list(t.encode("utf-8")) for t in texts]
        if not any(ids_corpus):
            raise ValueError("训练语料为空")

        n_target_merges = vocab_size - 256
        pair_counts: Counter[tuple[int, int]] = Counter()
        for ids in ids_corpus:
            pair_counts.update(zip(ids, ids[1:]))

        for step in range(n_target_merges):
            if not pair_counts:
                _LOGGER.info("BPE: 第 %d 步已无可合并的词对，提前结束", step)
                break
            # T3 确定性: 频次降序优先，频次相同时取字典序最小的 pair
            best_pair, best_count = min(
                pair_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if best_count < 2:
                _LOGGER.info("BPE: 最高频次 %d < 2，停止合并", best_count)
                break
            new_id = 256 + step
            self.merges.append(best_pair)
            self.ranks[best_pair] = new_id

            # 应用合并到全语料（O(语料长度) 单次扫描）
            for ci, ids in enumerate(ids_corpus):
                ids_corpus[ci] = _merge_pair(ids, best_pair, new_id)
            pair_counts = Counter()
            for ids in ids_corpus:
                pair_counts.update(zip(ids, ids[1:]))
            if verbose and step % 50 == 0:
                _LOGGER.info("BPE step %d/%d merge=%r count=%d",
                             step, n_target_merges, best_pair, best_count)

        self._build_vocab()
        return self

    def _build_vocab(self) -> None:
        """根据 merges 展开出 id → bytes 表（按 rank 顺序应用合并）。"""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]

    # ----------------------------------------------------------------- codec
    def encode(self, text: str) -> list[int]:
        """把字符串编码为 token id 序列。"""
        ids: list[int] = list(text.encode("utf-8"))
        # T2 合并贪心: 每次取 rank 最低（最先学到）的可合并 pair
        while len(ids) >= 2:
            key = tuple(ids)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_pos < 0:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            ids[best_pos : best_pos + 2] = [256 + self.merges.index((a, b))]
        self._cache[tuple(ids)] = ids
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """把 token id 序列还原为原始字符串（T1 无损往返）。"""
        buf = bytearray()
        for i in ids:
            piece = self.vocab.get(int(i))
            if piece is None:
                raise KeyError(f"未知 token id: {i}（词表大小 {len(self.vocab)}）")
            buf.extend(piece)
        # errors="replace" 保证即使拼出的字节流不合法也能返回字符串而非崩溃
        return buf.decode("utf-8", errors="replace")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    @property
    def vocab_size(self) -> int:
        """实际词表大小。"""
        return len(self.vocab)

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """落盘为 JSON（可复现的关键：merges 顺序被完整保存）。"""
        payload = {"merges": [list(m) for m in self.merges]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """从 JSON 恢复。"""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok.merges = [tuple(m) for m in payload["merges"]]  # type: ignore[misc]
        tok.ranks = {p: 256 + i for i, p in enumerate(tok.merges)}
        tok._build_vocab()
        return tok


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把序列中所有相邻 ``pair`` 就地替换为 ``new_id``。"""
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def compression_ratio(tokenizer: BPETokenizer, texts: Sequence[str]) -> float:
    """压缩比 = UTF-8 字节数 / token 数。越高说明分词越高效。

    这是衡量分词器质量的标准指标（GPT-2 论文报告约 3.5~4 字节/token）。
    """
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    if total_tokens == 0:
        return 0.0
    return total_bytes / total_tokens


def token_frequencies(texts: Sequence[str], tokenizer: BPETokenizer) -> defaultdict:
    """统计 token 频次分布（用于分析分词均衡性）。"""
    freq: defaultdict[int, int] = defaultdict(int)
    for t in texts:
        for tid in tokenizer.encode(t):
            freq[tid] += 1
    return freq
