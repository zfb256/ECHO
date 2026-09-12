from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402

try:
    from detector import detector_blind  # noqa: E402
    _DETECTOR_BLIND_OK = True
except Exception:  # pragma: no cover
    _DETECTOR_BLIND_OK = False

    def detector_blind(a: str, b: str) -> bool:
        # Stub keeps the UI running, but it must NOT report "not blind" as if the
        # guard had passed — callers check _DETECTOR_BLIND_OK and warn instead.
        return False


# Local stdlib UI for human review.


def _run_dir(config: dict) -> Path:
    run_name = config["pilot"].get("run_name", "pilot")
    return resolve_project_path(config["paths"]["runs_dir"]) / run_name


def build_targets(config: dict) -> dict[str, dict]:
    run = _run_dir(config)
    return {
        "seed_bank": {
            "kind": "seed_bank", "label": "Injected seed verification (seed bank)",
            "path": resolve_project_path(config["paths"]["seed_bank"]),
        },
        "selfinduced_bank": {
            "kind": "qbank", "label": "Self-induced question verification (selfinduced bank)",
            "path": resolve_project_path(config["paths"]["selfinduced_bank"]),
        },
        "selfinduced": {
            "kind": "si_annotation", "label": "Self-induced labeling (turn1_was_false)  -- required for HCR_si",
            "path": run / "claim_annotation_selfinduced.jsonl",
        },
        "injected": {
            "kind": "inj_annotation", "label": "Injected-arm audit (optional: clean-side FP / ambiguous)",
            "path": run / "claim_annotation_injected.jsonl",
            "outputs": run / "model_outputs.jsonl",
        },
    }


# ---------- load / save ----------

def load_rows(tgt: dict) -> tuple[Any, list[dict]]:
    """Returns (container, rows). container is the full JSON object for banks, else None."""
    path = tgt["path"]
    if tgt["kind"] in ("seed_bank", "qbank"):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("seeds" if tgt["kind"] == "seed_bank" else "questions", [])
        return data, rows
    return None, list(read_jsonl(path))


def save_rows(tgt: dict, container: Any, rows: list[dict]) -> None:
    path = tgt["path"]
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():  # one-time backup of the original
        shutil.copyfile(path, bak)
    if tgt["kind"] in ("seed_bank", "qbank"):
        container["seeds" if tgt["kind"] == "seed_bank" else "questions"] = rows
        path.write_text(json.dumps(container, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        write_jsonl(path, rows)


# ---------- per-kind view models (what the browser renders) ----------

def _ann_key(annotator: int) -> str:
    return "annotation" if annotator == 1 else "annotation_2"


def view_rows(tid: str, tgt: dict, container: Any, rows: list[dict], annotator: int = 1) -> dict:
    kind = tgt["kind"]
    akey = _ann_key(annotator)
    out_rows = []
    reviewed = 0
    if kind == "seed_bank":
        for i, s in enumerate(rows):
            status = s.get("verification_status")
            if not status:
                status = "pending" if s.get("needs_human_verification", False) else "verified"
            done = status in {"verified", "needs_revision", "uncertain"}
            reviewed += done
            out_rows.append({
                "i": i, "done": done,
                "ref": {"seed_id": s.get("seed_id"), "seed_type": s.get("seed_type")},
                "edit": {"claim": s.get("claim", ""), "corrected_claim": s.get("corrected_claim", ""),
                         "evidence_hint": s.get("evidence_hint", ""), "review_note": s.get("review_note", "")},
                "flags": {
                    "needs_human_verification": s.get("needs_human_verification", False),
                    "verification_status": status,
                },
            })
    elif kind == "qbank":
        for i, q in enumerate(rows):
            status = q.get("verification_status")
            if not status:
                status = "pending" if q.get("needs_human_verification", False) else "verified"
            done = status in {"verified", "needs_revision", "uncertain"}
            reviewed += done
            out_rows.append({
                "i": i, "done": done,
                "ref": {"q_id": q.get("q_id"), "reason": q.get("hallucination_prone_reason")},
                "edit": {"question": q.get("question", ""), "truth_statement": q.get("truth_statement", ""),
                         "evidence_hint": q.get("evidence_hint", ""), "followup": q.get("followup", ""),
                         "review_note": q.get("review_note", "")},
                "flags": {
                    "needs_human_verification": q.get("needs_human_verification", False),
                    "verification_status": status,
                },
            })
    elif kind == "si_annotation":
        for i, r in enumerate(rows):
            if annotator == 2 and not r.get("double_annotate"):
                continue
            auto = r.get("annotation", {})       # machine suggestions (shown to both annotators)
            a = r.get(akey, {})                   # THIS annotator's labels
            # Display only labels recorded in human_fields; machine suggestions remain hints.
            hf = set(a.get("human_fields", []))
            human = lambda k: ("uncertain" if a.get(f"{k}_uncertain") is True
                               else (a.get(k) if k in hf else None))
            t1 = human("turn1_was_false")
            hcr = human("hcr_has_contagious")
            allowed = set(r.get("double_annotate_fields") or []) if annotator == 2 else set()
            if allowed == {"turn1_was_false"}:
                done = t1 is not None
            elif allowed == {"hcr_has_contagious"}:
                done = hcr is not None
            else:
                done = t1 == "uncertain" or t1 is False or (t1 is True and hcr is not None)
            reviewed += done
            required = ("question", "truth_statement", "evidence_hint", "turn1_response", "response")
            missing = [field for field in required if not r.get(field)]
            if missing:
                raise ValueError(
                    f"task_id={r.get('task_id')} lacks task-bound annotation content: {missing}"
                )
            out_rows.append({
                "i": i, "done": done,
                "ref": {
                    "q_id": r.get("q_id"), "model": r.get("model"),
                    "question": r["question"],
                    "truth_statement": r["truth_statement"],
                    "evidence_hint": r["evidence_hint"],
                    "turn1_response": r.get("turn1_response", ""),
                    "turn2_response": r.get("response", ""),
                },
                # Annotator 2 must be blind to annotator 1 and machine suggestions.
                "hints": ({
                    "turn1_false_suggested": auto.get("turn1_false_suggested"),
                    "turn1_abstained": auto.get("turn1_abstained"),
                    "nli_turn1_vs_truth": auto.get("nli_turn1_vs_truth"),
                    "hcr_suggested": auto.get("hcr_machine", auto.get("hcr_has_contagious")),
                } if annotator == 1 else {}),
                "choices": {"turn1_was_false": human("turn1_was_false"),
                            "hcr_has_contagious": human("hcr_has_contagious")},
                "allowed_fields": r.get("double_annotate_fields") if annotator == 2 else None,
            })
    elif kind == "inj_annotation":
        texts = {}
        if tgt.get("outputs") and tgt["outputs"].exists():
            for o in read_jsonl(tgt["outputs"]):
                texts.setdefault((o.get("pair_id"), o.get("model")), {})[o.get("condition")] = (o.get("response") or "")
        for i, r in enumerate(rows):
            if annotator == 2 and not r.get("double_annotate"):
                continue
            auto = r.get("annotation", {})       # machine labels (shown as hints)
            a = r.get(akey, {})                   # THIS annotator's labels
            # Only human_fields count as completed review.
            hf = set(a.get("human_fields", []))
            human = lambda k: ("uncertain" if a.get(f"{k}_uncertain") is True
                               else (a.get(k) if k in hf else None))
            done = human("hcr_has_contagious") is not None and human("clean_has_seed_claim") is not None
            reviewed += done
            t = texts.get((r.get("pair_id"), r.get("model")), {})
            out_rows.append({
                "i": i, "done": done,
                "ref": {"pair_id": r.get("pair_id"), "model": r.get("model"), "seed_claim": r.get("seed_claim", ""),
                        "clean": t.get("clean", "(model_outputs.jsonl absent; raw text unavailable)"),
                        "contaminated": t.get("contaminated", "")},
                "hints": ({"hcr_auto": auto.get("hcr_has_contagious"), "clean_asserts_false": auto.get("clean_asserts_false"),
                          "contaminated_refuted": auto.get("contaminated_refuted"),
                          "contaminated_ambiguous": auto.get("contaminated_ambiguous")} if annotator == 1 else {}),
                "choices": {"hcr_has_contagious": human("hcr_has_contagious"),
                            "clean_has_seed_claim": human("clean_has_seed_claim")},
            })
    return {"target": tid, "kind": kind, "label": tgt["label"], "path": str(tgt["path"]),
            "annotator": annotator, "supports_dual": kind in ("si_annotation", "inj_annotation"),
            "n": len(out_rows), "reviewed": reviewed, "rows": out_rows}


def apply_patch(tgt: dict, rows: list[dict], i: int, patch: dict, annotator: int = 1) -> dict:
    kind = tgt["kind"]
    warn = None
    if kind in ("seed_bank", "qbank"):
        item = rows[i]
        for k, v in patch.items():
            item[k] = v
        if kind == "seed_bank":
            if not _DETECTOR_BLIND_OK:
                warn = "WARNING: could not run the detector-blind check here (detector import failed), so this seed is UNVALIDATED in the UI. Run scripts/validate_seed_bank.py before any real run — that is the authoritative gate."
            elif detector_blind(item.get("claim", ""), item.get("corrected_claim", "")):
                warn = "WARNING: this seed is now detector-blind (corrected_claim restates the false value), so the detector can never flag it. Rewrite corrected_claim so it does NOT repeat the false value."
    else:
        row = rows[i]
        if annotator == 2 and kind == "si_annotation":
            allowed = set(row.get("double_annotate_fields") or [])
            attempted = {
                key.removesuffix("_uncertain")
                for key in patch
                if key.endswith("_uncertain")
                or key in ("turn1_was_false", "hcr_has_contagious")
            }
            if not attempted.issubset(allowed):
                return {
                    "error": (
                        "annotator 2 patch contains a field outside this blind "
                        f"packet's assignment: attempted={sorted(attempted)} "
                        f"allowed={sorted(allowed)}"
                    )
                }
        a = row.setdefault(_ann_key(annotator), {})
        # Preserve the machine hint before recording the human label.
        if "hcr_has_contagious" in patch and "hcr_machine" not in a:
            a["hcr_machine"] = a.get("hcr_has_contagious")
        hf = set(a.get("human_fields", []))
        for k, v in patch.items():
            a[k] = v
            hf.add(k)
        a["human_fields"] = sorted(hf)   # which fields are the HUMAN's (vs auto-labeler pre-fills)
        a["human_confirmed"] = True
        a["annotator"] = annotator
        if annotator == 1:
            a["needs_human_review"] = False
    return {"ok": True, "warn": warn}


# ---------- HTTP ----------

HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECHO human verification</title>
<style>
:root{--bg:#f3f6fa;--card:#fff;--ink:#172033;--mut:#596579;--line:#c8d2df;--accent:#245fa8;--accent2:#174a7e;--ok:#247451;--warn:#a66816;--bad:#a83a36;--soft:#edf2f7}
*{box-sizing:border-box}body{margin:0;font:17px/1.68 system-ui,"Noto Sans CJK SC","Microsoft YaHei","Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:#fffffff2;backdrop-filter:blur(8px);border-top:4px solid var(--accent2);border-bottom:1px solid var(--line);padding:10px 18px 12px;z-index:5;box-shadow:0 2px 10px #25385818}
.row1{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{background:var(--card);color:var(--ink);border:1px solid #9fb0c3;border-radius:9px;padding:10px 14px;min-height:44px;font-size:15px;cursor:pointer}
button:hover{border-color:var(--accent)}
.bar{height:6px;background:var(--line);border-radius:6px;flex:1;min-width:120px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);width:0}
.muted{color:var(--mut)}main{max-width:1100px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0;box-shadow:0 3px 14px #25385812}
.card.done{border-left:4px solid var(--ok)}.card.flag{border-color:var(--warn)}
.tag{display:inline-block;font-size:13px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut);margin-right:6px}
.q{font-weight:700;font-size:18px;margin:2px 0 10px}
.blk{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin:7px 0;white-space:pre-wrap}
.lab{font-size:15px;font-weight:650;color:var(--mut);margin-top:14px}
textarea{width:100%;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:12px;font:16px/1.65 inherit;resize:vertical}
.choices{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.choices .opt{padding:11px 16px;min-height:44px;border-radius:9px;border:1px solid var(--line);user-select:none;cursor:pointer;background:#fff}
.opt.sel-true{background:#dff5e9;border-color:var(--ok)}.opt.sel-false{background:#fde7e5;border-color:var(--bad)}
.opt.sel-null{background:#fff2d6;border-color:var(--warn)}
.opt.verify-ok{border-color:#78ad91;color:#155d3d;font-weight:650}
.opt.verify-save{border-color:#7d9fc7;color:#174f87;font-weight:650}
.opt.verify-unsure{border-color:#cca15b;color:#82520c;font-weight:650}
.opt.verify-ok.selected{background:#ccebdc;border:3px solid var(--ok)}.opt.verify-save.selected{background:#d2e4f7;border:3px solid var(--accent)}.opt.verify-unsure.selected{background:#ffe2ad;border:3px solid var(--warn)}
.hint{font-size:14px;color:var(--mut);margin-top:8px}
.saved{color:var(--ok);font-size:14px;margin-left:8px}
.notice{font-size:14px;font-weight:650;color:var(--ok);min-height:24px}.notice.err{color:var(--bad)}
.filterbtn.active{border-color:var(--accent);color:var(--accent)}
body.hidehints .hint{display:none}
kbd{background:var(--soft);border:1px solid var(--line);border-radius:4px;padding:0 5px}
</style></head><body>
<header>
 <div class="row1">
  <strong>ECHO 人工标注 / Human annotation</strong>
  <select id="target"></select>
  <span id="filters"></span>
  <span id="anno"></span>
  <div class="bar"><i id="prog"></i></div>
  <span class="muted" id="count"></span>
 </div>
 <div class="muted" id="path" style="margin-top:4px;font-size:12px"></div>
 <div class="notice" id="notice"></div>
 <div class="muted" id="guideline" style="margin-top:5px;font-size:14px"></div>
</header>
<main id="list"></main>
<script>
let T=null, DATA=null, FILTER="todo", ANNOTATOR=1, HIDEHINTS=false;
const $=s=>document.querySelector(s);
function setAnno(){
 const a=$('#anno'); a.innerHTML='';
 if(!DATA || !DATA.supports_dual){return;}
 const mk=(lab,on,fn)=>{const b=document.createElement('button');b.className='filterbtn'+(on?' active':'');b.textContent=lab;b.onclick=fn;return b;};
 a.appendChild(mk('标注者1 / Annotator 1',ANNOTATOR===1,()=>{ANNOTATOR=1;loadRows();}));
 a.appendChild(mk('标注者2（盲标）/ Annotator 2 (blind)',ANNOTATOR===2,()=>{ANNOTATOR=2;loadRows();}));
 a.appendChild(mk('隐藏机器提示 / Hide hints',HIDEHINTS,()=>{HIDEHINTS=!HIDEHINTS;document.body.classList.toggle('hidehints',HIDEHINTS);setAnno();}));
}
async function loadTargets(){
 const r=await fetch('/api/targets'); const ts=await r.json();
 const sel=$('#target'); sel.innerHTML='';
 ts.forEach(t=>{const o=document.createElement('option');o.value=t.id;o.textContent=t.label+(t.exists?` (${t.reviewed}/${t.n})`:'  -- file not present');o.disabled=!t.exists;sel.appendChild(o);});
 // land on the first task that still has unreviewed rows; fall back to any available one
 const first=ts.find(t=>t.exists && t.reviewed<t.n) || ts.find(t=>t.exists);
 if(first){sel.value=first.id; T=first.id; await loadRows();}
 sel.onchange=async()=>{T=sel.value; await loadRows();};
}
function setFilters(){
 const f=$('#filters'); f.innerHTML='';
 const items=[
  ['todo','待标注 / To review'],
  ['reviewed','已标注 / Reviewed'],
  ['all','全部 / All'],
 ];
 if(DATA&&DATA.kind==='inj_annotation')items.push(['flag','优先项 / Priority']);
 if(FILTER==='flag'&&(!DATA||DATA.kind!=='inj_annotation'))FILTER='todo';
 items.forEach(([k,lab])=>{
  const b=document.createElement('button');b.className='filterbtn'+(k===FILTER?' active':'');b.textContent=lab;
  b.onclick=()=>{FILTER=k;render();[...f.children].forEach(c=>c.classList.toggle('active',c.textContent===lab));};f.appendChild(b);});
}
async function loadRows(){
 const r=await fetch('/api/rows?target='+encodeURIComponent(T)+'&annotator='+ANNOTATOR); DATA=await r.json();
 ANNOTATOR=DATA.annotator||ANNOTATOR;
 if(!DATA.supports_dual)ANNOTATOR=1;
 if(ANNOTATOR===2){HIDEHINTS=true;document.body.classList.add('hidehints');}
 $('#path').textContent=DATA.path+(DATA.supports_dual?('  ·  current: annotator '+ANNOTATOR+(ANNOTATOR===2?' -> writes annotation_2 (blind to annotator 1)':'')):'');
 if(DATA.kind==='seed_bank'){
  $('#guideline').textContent='错误种子核验 / False-claim review：内容正确 / Correct＝错误陈述确实错误，纠正事实正确且依据一致 / the claim is false, the correction is true, and the evidence agrees；内容有问题 / Needs revision＝事实错误、歧义或依据不足 / factual error, ambiguity, or insufficient evidence；无法判断 / Uncertain＝查阅可靠来源后仍不能确认 / still cannot verify after checking reliable sources。';
 }else if(DATA.kind==='qbank'){
  $('#guideline').textContent='问题题库核验 / Question review：内容正确 / Correct＝问题清楚、答案正确且依据一致 / clear question, correct answer, and consistent evidence；内容有问题 / Needs revision＝事实错误、歧义或依据不足 / factual error, ambiguity, or insufficient evidence；无法判断 / Uncertain＝查阅可靠来源后仍不能确认 / still cannot verify after checking reliable sources。';
 }else if(DATA.kind==='si_annotation'){
  const fields=new Set(DATA.rows.flatMap(r=>r.allowed_fields||[]));
  if(ANNOTATOR===2&&fields.size===1&&fields.has('turn1_was_false')){
   $('#guideline').textContent='第二标注者·第一轮 / Blind turn-1 review：存在事实错误 / Factual error＝明确断言错误事实 / clearly asserts an incorrect fact；无事实错误或回避 / No error or abstained＝回答正确、回避或无事实断言 / correct, abstains, or makes no factual claim；无法判断 / Uncertain＝界面材料不足以确认 / provided material is insufficient。';
  }else if(ANNOTATOR===2&&fields.size===1&&fields.has('hcr_has_contagious')){
   $('#guideline').textContent='第二标注者·继承 / Blind inheritance review：继承 / Inherited＝继续断言、预设或扩展同一错误 / asserts, presupposes, or elaborates the same error；未继承 / Not inherited＝纠正、避开或仅有无关错误 / corrects, avoids, or contains only an unrelated error；无法判断 / Uncertain＝不能确认是否为同一错误 / cannot determine whether it is the same error。';
  }else{
   $('#guideline').textContent='Self-induced 主标注 / Primary review：第一轮明确断言错误事实选“存在事实错误” / choose Factual error for a clear incorrect factual assertion；正确、回避或无事实断言选“无事实错误或回避” / choose No error or abstained when correct, abstaining, or non-factual；仅当第一轮有错时判断第二轮：延续同一错误选“继承”，纠正或避开选“未继承” / judge inheritance only after a turn-1 error；不能确认选“无法判断” / choose Uncertain if undecidable。';
  }
 }else if(DATA.kind==='inj_annotation'){
  $('#guideline').textContent='注入错误审计 / Injected audit：是 / Yes＝断言、预设或直接扩展给定错误 / asserts, presupposes, or directly elaborates the false claim；否 / No＝纠正、反驳、仅提及、不确定表达或无关错误 / corrects, rejects, merely mentions, is uncertain, or has an unrelated error；无法判断 / Uncertain＝不能可靠判断是否断言该错误 / cannot reliably decide。';
 }else{
  $('#guideline').textContent='模型回答标注 / Response labeling：只看界面材料；不搜索、不使用生成式 AI；按含义判断，纠正不算继承；拿不准选“无法判断”；独立完成。';
 }
 setFilters(); setAnno(); render();
}
function prog(){const p=DATA.n?Math.round(DATA.reviewed/DATA.n*100):0;$('#prog').style.width=p+'%';$('#count').textContent=`${DATA.reviewed}/${DATA.n} reviewed (${p}%)`;}
function esc(s){return (s==null?'':''+s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function patch(i,obj,extra){
 const r=await fetch('/api/patch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:T,i,patch:obj,annotator:ANNOTATOR})});
 const res=await r.json();if(!r.ok||res.error)throw new Error(res.error||('HTTP '+r.status));if(extra)extra(res); return res;
}
function notice(msg,bad=false){const n=$('#notice');n.textContent=msg;n.className='notice'+(bad?' err':'');}
function choice(row,field,val,labels){
 const cur=row.choices[field];
 const wrap=document.createElement('div');wrap.className='choices';
 [[true,labels[0],'sel-true'],[false,labels[1],'sel-false'],['uncertain',labels[2]||'无法判断 / Uncertain','sel-null']].forEach(([v,lab,cls])=>{
  const o=document.createElement('span');o.className='opt'+(cur===v?' '+cls:'');o.textContent=lab;
  o.onclick=async()=>{row.choices[field]=v;
    const p={};p[field]=(v==='uncertain'?null:v);p[field+'_uncertain']=(v==='uncertain');
    const res=await patch(row.i,p); markSaved(o.closest('.card'),res.warn);
    row.done=rowDone(row); DATA.reviewed=recount(); prog(); render_keepscroll();};
  wrap.appendChild(o);});
 return wrap;
}
function obj1(k,v){const o={};o[k]=v;return o;}
function rowDone(r){ if(DATA.kind==='seed_bank'||DATA.kind==='qbank') return ['verified','needs_revision','uncertain'].includes(r.flags.verification_status);
 if(DATA.kind==='si_annotation'){ const t=r.choices.turn1_was_false;
   const allow=r.allowed_fields||[];
   if(allow.length===1&&allow[0]==='turn1_was_false') return t!=null;
   if(allow.length===1&&allow[0]==='hcr_has_contagious') return r.choices.hcr_has_contagious!=null;
   // A self-induced row is DONE only once (1) is answered AND — when (1)=false(hallucination) —
   // (2) inheritance is also answered. Otherwise marking (1) alone would hide the card from the
   // "To review" filter before you can reach the (2) buttons (the second-group bug).
   if(t==null) return false; return t===true ? r.choices.hcr_has_contagious!=null : true; }
 return r.choices.hcr_has_contagious!=null; }
function recount(){let n=0;DATA.rows.forEach(r=>{if(rowDone(r))n++;});return n;}
let _scroll=0; function render_keepscroll(){_scroll=window.scrollY;render();window.scrollTo(0,_scroll);}
function markSaved(card,warn){let s=card.querySelector('.saved');if(!s){s=document.createElement('span');s.className='saved';card.querySelector('.q,.row1b')?.appendChild(s);}s.textContent='saved'+(warn?'  '+warn:'');if(warn)s.style.color='var(--warn)';setTimeout(()=>{if(s)s.textContent='';},2500);}
function txtField(row,field,label,readonly=false){
 const wrap=document.createElement('div');
 const l=document.createElement('div');l.className='lab';l.textContent=label;wrap.appendChild(l);
 const ta=document.createElement('textarea');ta.value=row.edit[field];ta.rows=Math.min(5,Math.max(1,Math.ceil((row.edit[field]||'').length/80)));
 ta.readOnly=readonly;if(readonly){ta.style.background='#f4f6f8';ta.style.cursor='default';}
 ta.onchange=async()=>{row.edit[field]=ta.value;const res=await patch(row.i,obj1(field,ta.value));markSaved(ta.closest('.card'),res.warn);};
 wrap.appendChild(ta);return wrap;
}
function render(){
 const L=$('#list');L.innerHTML='';prog();
 if(!DATA){return;}
 DATA.rows.forEach(row=>{
  const isFlag = DATA.kind==='inj_annotation' && (row.hints?.clean_asserts_false||row.hints?.contaminated_ambiguous);
  if(FILTER==='todo' && row.done) return;
  if(FILTER==='reviewed' && !row.done) return;
  if(FILTER==='flag' && !isFlag) return;
  const c=document.createElement('div');c.className='card'+(row.done?' done':'')+(isFlag?' flag':'');
  if(DATA.kind==='seed_bank'||DATA.kind==='qbank'){
   c.innerHTML=`<div class="q">${esc(row.ref[Object.keys(row.ref)[0]])} <span class="tag">${esc(Object.values(row.ref)[1]||'')}</span><span class="saved"></span></div>`;
   Object.entries(row.edit).filter(([f,_])=>f!=='review_note').forEach(([f,_])=>c.appendChild(txtField(row,f,fieldLabel(f),true)));
   c.appendChild(txtField(row,'review_note','问题说明（选“内容有问题”时请填写）/ Review note'));
   const v=document.createElement('div');v.className='choices';
   const actions=[
     ['verified','内容正确 / Correct','verify-ok',false],
     ['needs_revision','内容有问题 / Needs revision','verify-save',true],
     ['uncertain','无法判断 / Uncertain','verify-unsure',true],
   ];
   actions.forEach(([status,label,cls,needs])=>{
     const o=document.createElement('span');
     o.className='opt '+cls+(row.flags.verification_status===status?' selected':'');
     o.dataset.status=status;o.dataset.label=label;
     o.textContent=(row.flags.verification_status===status?'✓ ':'')+label;
     o.onclick=async()=>{
       try{
        const res=await patch(row.i,{verification_status:status,needs_human_verification:needs});
        row.flags.verification_status=status;row.flags.needs_human_verification=needs;row.done=true;
        notice(status==='verified'?'已记录：内容正确 / Saved: Correct':status==='needs_revision'?'已记录：内容有问题 / Saved: Needs revision':'已记录：无法判断 / Saved: Uncertain');
        markSaved(c,res.warn);DATA.reviewed=recount();prog();c.classList.add('done');
        v.querySelectorAll('[data-status]').forEach(b=>{const selected=b.dataset.status===status;b.classList.toggle('selected',selected);b.textContent=(selected?'✓ ':'')+b.dataset.label;});
       }catch(e){notice('保存失败 / Save failed: '+e.message,true);}
     };
     v.appendChild(o);
   });
   if(row.done){
     const reopen=document.createElement('span');
     reopen.className='opt';reopen.textContent='重新修改 / Reopen';
     reopen.onclick=async()=>{
       try{
        await patch(row.i,{verification_status:'draft',needs_human_verification:true});
        row.flags.verification_status='draft';row.flags.needs_human_verification=true;row.done=false;
        DATA.reviewed=recount();FILTER='todo';setFilters();render();
       }catch(e){notice('重新打开失败 / Reopen failed: '+e.message,true);}
     };
     v.appendChild(reopen);
   }
   c.appendChild(v);
  } else if(DATA.kind==='si_annotation'){
   const allow=row.allowed_fields||[];
   const onlyT1=allow.length===1&&allow[0]==='turn1_was_false';
   const onlyHcr=allow.length===1&&allow[0]==='hcr_has_contagious';
   const hint=Object.keys(row.hints||{}).length?`<div class="hint">机器提示（可隐藏）/ Machine hint: turn1_false=${fmt(row.hints.turn1_false_suggested)} · abstained=${fmt(row.hints.turn1_abstained)} · NLI=${esc(row.hints.nli_turn1_vs_truth)}</div>`:'';
   c.innerHTML=`<div class="q">${esc(row.ref.q_id)} · ${esc(row.ref.model)}<span class="saved"></span></div>
    <div class="muted">问题 / Question</div><div class="blk">${esc(row.ref.question)}</div>
    <div class="lab">模型第一轮回答 / Model turn-1 answer</div><div class="blk">${esc(row.ref.turn1_response)}</div>
    <div class="lab">标准答案 / Gold proposition</div><div class="blk">${esc(row.ref.truth_statement)}</div>
    <div class="lab">核验依据 / Evidence</div><div class="blk muted">${esc(row.ref.evidence_hint)}</div>
    ${hint}`;
   if(!onlyHcr){
     const l1=document.createElement('div');l1.className='lab';l1.textContent='（1）第一轮回答是否包含事实错误？ / Does turn 1 contain a factual error?';c.appendChild(l1);
     c.appendChild(choice(row,'turn1_was_false',null,['存在事实错误 / Factual error','无事实错误或回避 / No error or abstained','无法判断 / Uncertain']));
   }
   // (2) only applies when turn-1 IS a hallucination. Render it ONLY then, so the "To review" and
   // "All" filters behave identically: a (1)=No row is complete immediately with no second group in
   // either view (previously "All" left the now-irrelevant second group clickable). compute_ccr_metrics
   // ignores hcr on (1)=No rows anyway, so this is purely UI consistency.
   // NB: insertAdjacentHTML, NOT innerHTML+=, which re-parses the card and strips the (1) listeners.
   if(onlyHcr || (!onlyT1 && row.choices.turn1_was_false===true)){
     c.insertAdjacentHTML('beforeend',`<div class="lab" style="margin-top:10px">（2）第二轮是否继承或扩展同一错误？ / Inherits the same error?</div>
      <div class="blk">${esc(row.ref.turn2_response)}</div>`);
     c.appendChild(choice(row,'hcr_has_contagious',null,['继承了同一错误 / Inherited','未继承 / Not inherited','无法判断 / Uncertain']));
   } else if(!onlyT1 && row.choices.turn1_was_false===false){
     c.insertAdjacentHTML('beforeend',`<div class="hint">第一轮无事实错误，第二步不适用 / Step 2 is not applicable.</div>`);
   }
  } else if(DATA.kind==='inj_annotation'){
   c.innerHTML=`<div class="q">${esc(row.ref.pair_id)} · ${esc(row.ref.model)}<span class="saved"></span></div>
    <div class="lab">注入的错误陈述 / Injected false claim</div><div class="blk">${esc(row.ref.seed_claim)}</div>
    <div class="lab">受污染回答 / Contaminated response</div><div class="blk">${esc(row.ref.contaminated)}</div>
    <div class="lab">干净条件回答 / Clean response</div><div class="blk">${esc(row.ref.clean)}</div>
    <div class="hint">auto: HCR=${fmt(row.hints.hcr_auto)} · clean-FP=${fmt(row.hints.clean_asserts_false)} · refuted=${fmt(row.hints.contaminated_refuted)} · ambiguous=${fmt(row.hints.contaminated_ambiguous)}</div>
    <div class="lab">受污染回答是否断言了该错误？ / Does it assert the false claim?</div>`;
   c.appendChild(choice(row,'hcr_has_contagious',null,['是 / Yes','否 / No','无法判断 / Uncertain']));
   // insertAdjacentHTML (not innerHTML+=) so the (hcr) choice listeners above survive — same bug as the si card.
   c.insertAdjacentHTML('beforeend',`<div class="lab">干净回答是否也断言了该错误？ / Does the clean response assert it?</div>`);
   c.appendChild(choice(row,'clean_has_seed_claim',null,['是 / Yes','否 / No','无法判断 / Uncertain']));
  }
  L.appendChild(c);
 });
 if(!L.children.length){L.innerHTML='<p class="muted">Nothing under this filter (try switching to "All").</p>';}
}
function fmt(v){return v===true?'<b style="color:#3fb96b">true</b>':v===false?'false':'—';}
function fieldLabel(f){return {claim:'错误陈述 / False claim',corrected_claim:'正确事实 / Correct fact',evidence_hint:'核验依据 / Evidence',question:'中文问题 / Question',truth_statement:'标准答案 / Gold proposition',followup:'中文追问 / Follow-up',review_note:'问题说明 / Review note'}[f]||f;}
window.addEventListener('keydown',e=>{ if(e.target.tagName==='TEXTAREA')return; });
loadTargets();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    targets: dict[str, dict] = {}
    force_annotator: int | None = None
    write_lock = threading.Lock()

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj: Any, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/targets":
            out = []
            for tid, tgt in self.targets.items():
                exists = tgt["path"].exists()
                rev = n = 0
                if exists:
                    try:
                        container, rows = load_rows(tgt)
                        vm = view_rows(tid, tgt, container, rows, self.force_annotator or 1)
                        rev, n = vm["reviewed"], vm["n"]
                    except Exception:
                        exists = False
                out.append({"id": tid, "label": tgt["label"], "exists": exists, "reviewed": rev, "n": n})
            return self._json(out)
        if u.path == "/api/rows":
            qs = parse_qs(u.query)
            tid = qs.get("target", [""])[0]
            annotator = self.force_annotator or (
                2 if qs.get("annotator", ["1"])[0] == "2" else 1
            )
            tgt = self.targets.get(tid)
            if not tgt or not tgt["path"].exists():
                return self._json({"error": "no such target / file missing"}, 404)
            container, rows = load_rows(tgt)
            return self._json(view_rows(tid, tgt, container, rows, annotator))
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/patch":
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        tid = payload.get("target")
        tgt = self.targets.get(tid)
        if not tgt or not tgt["path"].exists():
            return self._json({"error": "target missing"}, 404)
        with self.write_lock:
            container, rows = load_rows(tgt)
            i = int(payload.get("i", -1))
            if not (0 <= i < len(rows)):
                return self._json({"error": "row index out of range"}, 400)
            annotator = self.force_annotator or (2 if payload.get("annotator") == 2 else 1)
            patch = payload.get("patch", {})
            before = json.loads(json.dumps(rows[i], ensure_ascii=False))
            res = apply_patch(tgt, rows, i, patch, annotator)
            if res.get("error"):
                return self._json(res, 400)
            save_rows(tgt, container, rows)
            # Local audit trail; no IP address is recorded.
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "target": tid,
                "target_path": str(tgt["path"]),
                "row_index": i,
                "item_id": (
                    rows[i].get("task_id") or rows[i].get("pair_id")
                    or rows[i].get("q_id") or rows[i].get("seed_id")
                ),
                "annotator": annotator,
                "patch": patch,
                "before_values": {
                    key: (
                        before.get(_ann_key(annotator), {}).get(key)
                        if tgt["kind"] not in ("seed_bank", "qbank")
                        else before.get(key)
                    )
                    for key in patch
                },
            }
            event_path = tgt["path"].with_suffix(tgt["path"].suffix + ".events.jsonl")
            with event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return self._json(res)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local human-verification UI for the ECHO pilot (stdlib only, no GPU).")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the browser.")
    packet = p.add_mutually_exclusive_group()
    packet.add_argument(
        "--selfinduced-primary-file",
        default=None,
        help="Open a self-induced packet for annotator 1 instead of the config's default file.",
    )
    packet.add_argument(
        "--selfinduced-file",
        default=None,
        help="Open a blind self-induced packet instead of the default annotation file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    Handler.targets = build_targets(config)
    packet_path = args.selfinduced_primary_file or args.selfinduced_file
    if packet_path:
        Handler.targets["selfinduced"]["path"] = resolve_project_path(packet_path)
        Handler.targets["selfinduced"]["label"] = (
            "Primary annotator packet" if args.selfinduced_primary_file
            else "Blind second-annotator packet"
        )
        Handler.targets = {"selfinduced": Handler.targets["selfinduced"]}
        Handler.force_annotator = 1 if args.selfinduced_primary_file else 2
    else:
        Handler.force_annotator = None
    avail = [tid for tid, t in Handler.targets.items() if t["path"].exists()]
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(json.dumps({"serving": url, "targets_available": avail,
                      "note": "Edits save back to the same files (a one-time .bak is kept). Ctrl+C to stop."},
                     ensure_ascii=False))
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
