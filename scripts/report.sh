#!/bin/bash
# tables and figures of the report from the evaluation summaries --> report/.
# the llm-judge tables are added when judge dirs are given in MALD_JUDGE
# (colon-separated; see README).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; MPL=${MPL:-$PY}
R=${MALD_REPORT:-report}; mkdir -p "$R"
V="m_full=Agentic,_LLM_meta-review det_t0.3_w0=Ours_(agentic),_selected det_nemo=Agentic,_Nemotron m_full_nemo=Agentic,_LLM_meta-review,_Nemotron m_unc=Agentic,_LLM_meta-review,_no_rebuttal-mode m_noret=Agentic,_LLM_meta-review,_no_retrieval m_flags=Agentic,_LLM_meta-review,_theta_0.3 m_nomodes=Agentic,_LLM_meta-review,_no_mode_tags m_norubric=Agentic,_LLM_meta-review,_no_scores m_noflags=Agentic,_LLM_meta-review,_neither b1=Ours b1_nemo=Ours,_Nemotron b1old=ML_baseline b0=Non-ML_baseline"
T="b0=Non-ML_baseline b1old=ML_baseline b1=Ours m=Ours_(agentic) m_unc=Agentic,_no_rebuttal-mode m_alt=Agentic,_LLM_meta-review"
X="b1=Ours m=Ours_(agentic) b1_nemotron=Ours,_Nemotron m_nemotron=Agentic,_Nemotron m_mix2=Agentic,_mixed_panel b1_qwen=Ours,_Qwen m_qwen=Agentic,_Qwen"
P="m=Ours_(agentic) b1=Ours b1old=ML_baseline b0=Non-ML_baseline"
PV="det_t0.3_w0=Ours_(agentic) b1=Ours b1old=ML_baseline b0=Non-ML_baseline"
RT="m=Ours_(agentic) b1=Ours b0=Non-ML_baseline m_unc=Agentic,_no_rebuttal-mode m_alt=Agentic,_LLM_meta-review"
$PY -m mald.table val $V > "$R/val_tables.tex"
$MPL -m mald.plot val "$R/val_curves.pdf" $PV
$PY -m mald.table test $T > "$R/test_tables.tex"
$MPL -m mald.plot test "$R/test_curves.pdf" $P
$PY -m mald.table test $X > "$R/test_tables_models.tex"
$PY -m mald.rebut test $RT > "$R/rebut_test.tex"
$PY -m mald.extra > /dev/null
if [ -n "${MALD_JUDGE:-}" ]; then
  $PY -m mald.judge_agg > "$R/judge_tables.tex"
  $MPL -m mald.plot judge "${MALD_JUDGE%%:*}/agg.json" "$R/judge_curves.pdf" $P
fi
ls "$R"
