"""
Recall Debug 模块 - 用于追踪指定目标零件在三路融合检索各阶段的状

通过 RecallDebugger 类，可以记录目标零件在 text_recall / geo_recall / visual_recall /
merge / fusion / final 各阶段的状态，并自动诊断丢失原因。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class StageRecord:
    """单个阶段的记录"""

    def __init__(self, name: str):
        self.name = name
        self.present: bool = False          # 目标是否在该阶段出现
        self.rank: Optional[int] = None     # 在该阶段的排名
        self.score: Optional[float] = None  # 在该阶段的分数
        self.total: int = 0                 # 该阶段候选总数
        self.detail: str = ""               # 额外描述

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "rank": self.rank,
            "score": self.score,
            "total": self.total,
            "detail": self.detail,
        }


class RecallDebugger:
    """
    追踪目标零件在各检索阶段的状态，输出诊断报告。

    Usage:
        debugger = RecallDebugger("5158F-QJ753-00002.stp")
        # 在各阶段调用 record_* 方法
        debugger.record_text_recall(candidates, ...)
        debugger.record_geo_recall(candidates, ...)
        ...
        debugger.print_report()
    """

    def __init__(self, target_filename: str):
        self.target_filename = target_filename
        self.records: Dict[str, StageRecord] = {}
        self.found: bool = False  # 目标是否在最终结果中

    # ---- 记录各阶段 ----

    def record_text_recall(self, candidates: List[dict], detail: str = ""):
        """记录文本召回阶段"""
        rec = StageRecord("text_recall")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("text_similarity", c.get("similarity", 0))
                break
        rec.total = len(candidates)
        rec.detail = detail or f"文本召回 {len(candidates)} 个候选"
        self.records["text_recall"] = rec

    def record_geo_recall(self, candidates: List[dict], detail: str = ""):
        """记录几何召回阶段"""
        rec = StageRecord("geo_recall")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("geo_similarity", 0)
                break
        rec.total = len(candidates)
        rec.detail = detail or f"几何召回 {len(candidates)} 个候选"
        self.records["geo_recall"] = rec

    def record_visual_recall(self, candidates: List[dict], detail: str = ""):
        """记录视觉召回阶段"""
        rec = StageRecord("visual_recall")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("visual_similarity", 0)
                break
        rec.total = len(candidates)
        rec.detail = detail or f"视觉召回 {len(candidates)} 个候选"
        self.records["visual_recall"] = rec

    def record_merge(self, candidates: List[dict], detail: str = ""):
        """记录合并去重阶段"""
        rec = StageRecord("merge")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("recall_count", 0)
                rec.detail = f"被 {c.get('recall_count', 0)} 路召回，来源: {c.get('recall_sources', [])}"
                break
        rec.total = len(candidates)
        if not rec.detail:
            rec.detail = detail or f"合并去重 {len(candidates)} 个候选"
        self.records["merge"] = rec

    def record_fusion(self, candidates: List[dict], detail: str = ""):
        """记录融合排序阶段"""
        rec = StageRecord("fusion")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("fusion_score", 0)
                break
        rec.total = len(candidates)
        rec.detail = detail or f"融合排序 {len(candidates)} 个候选"
        self.records["fusion"] = rec

    def record_final(self, candidates: List[dict], detail: str = ""):
        """记录最终结果阶段"""
        rec = StageRecord("final")
        for i, c in enumerate(candidates):
            if self._match(c):
                rec.present = True
                rec.rank = i + 1
                rec.score = c.get("final_score", c.get("hybrid_similarity", c.get("similarity", 0)))
                self.found = True
                break
        rec.total = len(candidates)
        rec.detail = detail or f"最终结果 {len(candidates)} 个候选"
        self.records["final"] = rec

    # ---- 诊断 ----

    def diagnose(self) -> Dict[str, Any]:
        """自动诊断目标零件在哪个阶段丢失"""
        stages = ["text_recall", "geo_recall", "visual_recall", "merge", "fusion", "final"]
        diagnoses = []

        # 检查各阶段是否被记录
        recorded = [s for s in stages if s in self.records]
        if not recorded:
            return {"found": False, "diagnoses": ["未记录任何阶段数据，无法诊断"]}

        # 找到第一个丢失的阶段
        first_missing = None
        for s in stages:
            if s in self.records and not self.records[s].present:
                first_missing = s
                break

        if first_missing is None and self.found:
            diagnoses.append("目标零件在所有阶段均出现，最终结果中包含目标。")
            return {"found": True, "diagnoses": diagnoses}

        if first_missing is None and not self.found:
            # 所有阶段都有记录但最终没有——这通常不会发生，但做防御
            diagnoses.append("目标零件在召回阶段出现但最终结果中丢失（可能是排名过低被截断）。")
            return {"found": False, "diagnoses": diagnoses}

        # 根据第一个丢失的阶段生成诊断
        if first_missing == "text_recall":
            diagnoses.append(
                "【文本召回丢失】目标零件在文本Embedding召回阶段未被召回。\n"
                "  可能原因: 文本描述特征与目标零件不匹配，或Embedding模型对该零件表达能力不足。\n"
                "  建议: 检查文本描述生成是否准确，或尝试增大 text_recall_k 参数。"
            )
        elif first_missing == "geo_recall":
            diagnoses.append(
                "【几何召回丢失】目标零件在几何向量召回阶段未被召回。\n"
                "  可能原因: 几何特征向量无法有效区分该零件，或几何索引未正确构建。\n"
                "  建议: 检查几何索引是否包含该零件，或尝试增大 geo_recall_k 参数。"
            )
        elif first_missing == "visual_recall":
            diagnoses.append(
                "【视觉召回丢失】目标零件在视觉向量召回阶段未被召回。\n"
                "  可能原因: 视觉特征向量提取质量不足，或视图图片缺失/质量差。\n"
                "  建议: 检查视图图片是否存在和质量，或尝试增大 visual_recall_k 参数。"
            )
        elif first_missing == "merge":
            diagnoses.append(
                "【合并去重丢失】目标零件在合并去重阶段未被包含。\n"
                "  可能原因: 三路召回均未命中该零件。\n"
                "  建议: 检查各独立召回阶段的配置和索引覆盖。"
            )
        elif first_missing == "fusion":
            last_rec = self.records.get("merge")
            if last_rec and last_rec.present:
                recall_count = int(last_rec.score) if last_rec.score else 0
                diagnoses.append(
                    f"【融合排序丢失】目标零件在召回阶段出现（{recall_count}/3路），但融合排序后被过滤。\n"
                    "  可能原因: fusion_top_k 太小，或融合权重不合理导致该零件排名过低。\n"
                    "  建议: 增大 fusion_top_k，或调整融合权重使该零件获得更高分数。"
                )
            else:
                diagnoses.append(
                    "【融合排序丢失】目标零件在融合排序阶段未被包含。\n"
                    "  可能原因: 上一阶段已有该零件但排名过低被截断。"
                )
        elif first_missing == "final":
            last_rec = self.records.get("fusion")
            if last_rec and last_rec.present:
                diagnoses.append(
                    "【最终结果丢失】目标零件在融合排序阶段出现（排名 {}/{}），但最终结果中被过滤。\n"
                    "  可能原因: 几何精排后分数过低，或 top_k 太小。\n"
                    "  建议: 增大最终返回数量，或检查几何精排的评分逻辑。"
                )
            else:
                diagnoses.append(
                    "【最终结果丢失】目标零件在融合排序阶段未被包含。"
                )

        # 补充信息
        for s in stages:
            if s in self.records and self.records[s].present:
                r = self.records[s]
                diag = f"  [OK] {s}: 目标存在，排名 #{r.rank}/{r.total}"
                if r.score is not None:
                    diag += f"，分数={r.score:.4f}"
                diagnoses.append(diag)

        return {"found": self.found, "diagnoses": diagnoses}

    def print_report(self):
        """输出格式化的调试报告"""
        print("\n" + "=" * 60)
        print(f"  Recall Debug Report")
        print(f"  Target: {self.target_filename}")
        print("=" * 60)

        stages = [
            ("text_recall", "文本召回"),
            ("geo_recall", "几何召回"),
            ("visual_recall", "视觉召回"),
            ("merge", "合并去重"),
            ("fusion", "融合排序"),
            ("final", "最终结果"),
        ]

        for key, label in stages:
            rec = self.records.get(key)
            if rec is None:
                print(f"  [{label}] 未记录")
                continue

            status = "[OK]" if rec.present else "[NO]"
            print(f"  [{label}] {status}  ", end="")
            if rec.present:
                print(f"排名 #{rec.rank}/{rec.total}", end="")
                if rec.score is not None:
                    print(f"  分数={rec.score:.4f}", end="")
                print()
                if rec.detail:
                    print(f"          {rec.detail}")
            else:
                print(f"目标未出现 (共{rec.total}个候选)")
                if rec.detail:
                    print(f"          {rec.detail}")

        # 诊断
        print()
        diag = self.diagnose()
        print(f"  Diagnosis:")
        for d in diag["diagnoses"]:
            for line in d.split("\n"):
                print(f"    {line}")

        print("=" * 60)

    # ---- 内部辅助 ----

    def _match(self, candidate: dict) -> bool:
        """判断候选是否为目标零件"""
        # 尝试多种字段名
        for key in ("filename", "filepath", "path", "id"):
            val = candidate.get(key, "")
            if val and Path(val).name == self.target_filename:
                return True
        return False