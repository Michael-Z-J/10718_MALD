#!/bin/bash
# all experiments, in one sequential lane. a rerun skips finished papers.
#
# corpus --> gt (val, test) --> val sweep --> best det config --> flag variants
#   --> selection --> test: baselines, selected config, ablations, other
#   generators, mixed panel --> evaluation (test with the llm matcher) --> tables.
#
# needs: llama.cpp servers for the models in use (see README), or a full llm
# cache (MALD_CACHE) for a rerun without servers. QWEN=1 adds the qwen runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
NV=${NV:-12}; NT=${NT:-40}
R=${MALD_REPORT:-report}
run() { $PY -m mald.run "$@"; }
kw() { $PY -c "import json,sys;d=json.loads(sys.argv[1]);d.update(json.loads(sys.argv[2]));print(json.dumps(d))" "$1" "$2"; }

DETS=""
for t in 0.1 0.15 0.2 0.25 0.3; do for w in 0 0.5 1; do DETS="$DETS det_t${t}_w${w}"; done; done
VAL="m_full b1 b0 b1old m_unc m_noret b1_nemo m_full_nemo"
FLAGS="m_flags m_nomodes m_norubric m_noflags"

BUILD=$($PY -c "from mald.paths import BUILD; print(BUILD)")
[ -s "$BUILD/forums.parquet" ] || $PY -m mald.build
$PY -m mald.gt val $NV
$PY -m mald.gt test $NT

# val sweep
run val $NV m_full m '{"theta":0.2}'
run val $NV b1 b1 '{}'
run val $NV b0 b0 '{}'
run val $NV b1old b1old '{}'
run val $NV m_unc m '{"n_reb":0,"theta":0.2}'
run val $NV m_noret m '{"retrieval":false,"theta":0.2}'
run val $NV b1_nemo b1 '{"model":"nemotron"}'
run val $NV m_full_nemo m '{"model":"nemotron","theta":0.2}'
for d in $DETS; do
  t=${d#det_t}; t=${t%_w*}; w=${d##*_w}
  run val $NV $d m "{\"theta\":$t,\"w\":$w,\"llm_meta\":false}"
done
$PY -m mald.evaluate val nojudge $VAL $DETS > /dev/null
TW=$($PY -m mald.select det)
run val $NV det_nemo m "$(kw "$TW" '{"model":"nemotron","llm_meta":false}')"
run val $NV m_flags m "$TW"
run val $NV m_nomodes m "$(kw "$TW" '{"modes":false}')"
run val $NV m_norubric m "$(kw "$TW" '{"rubric":false}')"
run val $NV m_noflags m "$(kw "$TW" '{"modes":false,"rubric":false}')"
$PY -m mald.evaluate val nojudge $VAL $DETS det_nemo $FLAGS > /dev/null
$PY -m mald.select
SEL=$(cat "$R/selected.json")

# test
run test $NT b0 b0 '{}'
run test $NT b1old b1old '{}'
run test $NT b1 b1 '{}'
run test $NT m m "$SEL"
run test $NT m_unc m "$(kw "$SEL" '{"n_reb":0}')"
run test $NT m_alt m "$($PY -c "import json,sys;d=json.loads(sys.argv[1]);d['llm_meta']=not d.get('llm_meta',True);print(json.dumps(d))" "$SEL")"
B1=$(cat "$R/selected_b1.json")
[ "$B1" != "{}" ] && run test $NT b1_nemo b1 "$B1"
run test $NT b1_nemotron b1 '{"model":"nemotron"}'
run test $NT m_nemotron m "$(kw "$SEL" '{"model":"nemotron"}')"
run test $NT m_mix2 m "$(kw "$SEL" '{"models":["gemma","nemotron","gemma","nemotron","gemma","nemotron"]}')"
EXT="b1_nemotron m_nemotron m_mix2"
if [ "${QWEN:-0}" = 1 ]; then
  run test $NT b1_qwen b1 '{"model":"qwen"}'
  run test $NT m_qwen m "$(kw "$SEL" '{"model":"qwen"}')"
  EXT="$EXT b1_qwen m_qwen"
fi

# other generators first (no llm matcher), then the main systems with it:
# a nojudge evaluation of a system drops its llm-matcher columns.
MAIN="m b1 b0 b1old m_unc m_alt"; [ "$B1" != "{}" ] && MAIN="$MAIN b1_nemo"
$PY -m mald.evaluate test nojudge b1 $EXT > /dev/null
$PY -m mald.evaluate test judge $MAIN
scripts/report.sh
echo PIPELINE_DONE
