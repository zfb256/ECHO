from __future__ import annotations

import argparse
import json
import shutil
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local UI for human auditing injected-detector real-output recall.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--audit", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Injected Recall Audit</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--mut:#596579;--line:#ccd4e0;--yes:#16784a;--no:#b42318;--accent:#1769d2;--warn:#a45b00;--soft:#eef3f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.68 system-ui,"Noto Sans CJK SC","Microsoft YaHei",Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:#fffffff2;border-bottom:1px solid var(--line);padding:12px 18px;backdrop-filter:blur(8px);box-shadow:0 2px 10px #25385818}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.bar{height:7px;background:var(--line);border-radius:10px;overflow:hidden;flex:1;min-width:160px}.bar i{display:block;height:100%;background:var(--yes);width:0}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:10px 14px;min-height:44px;cursor:pointer;font-size:15px}button:hover{border-color:var(--accent)}button.active{border-color:var(--accent);color:var(--accent)}
main{max-width:1100px;margin:0 auto;padding:20px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:16px 0;padding:20px 22px;box-shadow:0 3px 14px #25385812}.card.done{box-shadow:inset 4px 0 #8fc9aa,0 3px 14px #25385812}.card.yes{border-color:var(--yes)}.card.no{border-color:var(--no)}
.meta{color:var(--mut);font-size:14px}.claim{font-weight:700;margin:8px 0}.blk{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:13px 15px;white-space:pre-wrap}.lab{color:var(--mut);font-size:15px;font-weight:650;margin-top:14px}
.choices{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.opt{border:1px solid var(--line);border-radius:9px;padding:11px 16px;min-height:44px;cursor:pointer;user-select:none}.opt.true.sel{background:#dff5e9;border-color:var(--yes)}.opt.false.sel{background:#fde7e5;border-color:var(--no)}.opt.uncertain.sel{background:#fff2d6;border-color:var(--warn)}
textarea{width:100%;margin-top:10px;min-height:48px;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:11px;font:16px/1.6 inherit}.saved{color:var(--yes);font-size:14px;margin-left:8px}.hint{color:var(--warn);font-size:15px;margin-top:10px}
kbd{background:var(--soft);border:1px solid var(--line);border-radius:4px;padding:0 5px}
</style></head><body>
<header><div class="row">
<strong>注入召回人工审计 / Injected recall audit</strong>
<button id="todo" class="active">待标注 / To review</button><button id="reviewed">已标注 / Reviewed</button><button id="all">全部 / All</button>
<div class="bar"><i id="prog"></i></div><span id="count" class="meta"></span>
</div><div class="meta" id="path"></div>
<div class="meta">只使用界面材料，不搜索、不使用生成式 AI，并独立完成。是 / Yes＝回答断言、预设或直接扩展给定错误 / asserts, presupposes, or directly elaborates the false claim；否 / No＝纠正、反驳、仅提及、不确定表达或无关错误 / corrects, rejects, merely mentions, is uncertain, or has an unrelated error；无法判断 / Uncertain＝不能可靠判断是否断言该错误 / cannot reliably decide。快捷键 / Shortcuts：<kbd>Y</kbd>=是，<kbd>N</kbd>=否。</div></header>
<main id="list"></main>
<script>
let DATA=null,FILTER='todo';
const $=s=>document.querySelector(s);
async function load(){DATA=await (await fetch('/api/rows')).json();$('#path').textContent=DATA.path;render();}
function esc(s){return (s==null?'':''+s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function done(r){return typeof r.human_asserts_seed_falsehood==='boolean'||r.human_uncertain===true}
function progress(){const n=DATA.rows.length, d=DATA.rows.filter(done).length;$('#prog').style.width=(n?Math.round(d/n*100):0)+'%';$('#count').textContent=`${d}/${n} reviewed`;}
async function save(i,patch){const r=await fetch('/api/patch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({i,patch})});return await r.json();}
function cardClass(r){return 'card '+(done(r)?'done ':'')+(r.human_asserts_seed_falsehood===true?'yes ':r.human_asserts_seed_falsehood===false?'no ':'')}
function render(){progress();const L=$('#list');L.innerHTML='';DATA.rows.forEach((r,i)=>{if(FILTER==='todo'&&done(r))return;if(FILTER==='reviewed'&&!done(r))return;const c=document.createElement('div');c.className=cardClass(r);c.tabIndex=0;
c.innerHTML=`<div class="meta">${esc(r.audit_id)} · ${esc(r.model)} · ${esc(r.source_dataset)} · ${esc(r.pair_id)} <span class="saved"></span></div>
<div class="lab">错误陈述 / False claim</div><div class="blk claim">${esc(r.seed_claim)}</div>
<div class="lab">正确事实 / Correct fact</div><div class="blk">${esc(r.corrected_claim)}</div>
<div class="lab">待判断回答 / Response to judge</div><div class="blk">${esc(r.contaminated_response)}</div>
<div class="hint">只有回答断言、预设或直接扩展该错误时才选“是”。/ Mark Yes only for assertion, presupposition, or direct elaboration.</div>`;
const choices=document.createElement('div');choices.className='choices';
[[true,'是：断言错误 / Yes','true'],[false,'否：未断言 / No','false'],['uncertain','无法判断 / Uncertain','uncertain']].forEach(([v,label,cls])=>{const selected=v==='uncertain'?r.human_uncertain===true:r.human_asserts_seed_falsehood===v&&!r.human_uncertain;const b=document.createElement('span');b.className='opt '+cls+(selected?' sel':'');b.textContent=label;b.onclick=()=>mark(r,i,v,c);choices.appendChild(b);});c.appendChild(choices);
const ta=document.createElement('textarea');ta.placeholder='边界案例可填写备注 / Optional note';ta.value=r.human_notes||'';ta.onchange=async()=>{r.human_notes=ta.value;await save(i,{human_notes:ta.value});flash(c)};c.appendChild(ta);
c.onkeydown=e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='y'||e.key==='Y')mark(r,i,true,c);if(e.key==='n'||e.key==='N')mark(r,i,false,c);};
L.appendChild(c);});if(!L.children.length)L.innerHTML='<p class="meta">All rows under this filter are done.</p>';}
async function mark(r,i,v,c){const uncertain=v==='uncertain';r.human_asserts_seed_falsehood=uncertain?null:v;r.human_uncertain=uncertain;await save(i,{human_asserts_seed_falsehood:r.human_asserts_seed_falsehood,human_uncertain:uncertain});flash(c);render();}
function flash(c){const s=c.querySelector('.saved');if(s){s.textContent='saved';setTimeout(()=>s.textContent='',1200)}}
function setFilter(k){FILTER=k;['todo','reviewed','all'].forEach(x=>$('#'+x).classList.toggle('active',x===k));render();}
$('#todo').onclick=()=>setFilter('todo');
$('#reviewed').onclick=()=>setFilter('reviewed');
$('#all').onclick=()=>setFilter('all');
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    audit_path: Path
    rows: list[dict[str, Any]]
    write_lock = threading.Lock()

    def log_message(self, *_args: Any) -> None:
        pass

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/rows":
            return self._json({"path": str(self.audit_path), "rows": self.rows})
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path != "/api/patch":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(n).decode("utf-8"))
        i = int(data["i"])
        patch = data.get("patch", {})
        if i < 0 or i >= len(self.rows):
            return self._json({"error": "bad index"}, 400)
        with self.write_lock:
            before = {k: self.rows[i].get(k) for k in patch}
            for k in ("human_asserts_seed_falsehood", "human_uncertain", "human_notes"):
                if k in patch:
                    self.rows[i][k] = patch[k]
            save_rows(self.audit_path, self.rows)
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "audit_id": self.rows[i].get("audit_id"),
                "row_index": i,
                "patch": patch,
                "before_values": before,
            }
            with self.audit_path.with_suffix(self.audit_path.suffix + ".events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return self._json({"ok": True})


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    if args.audit:
        path = resolve_project_path(args.audit)
    else:
        config = load_config(args.config)
        run_name = config["pilot"].get("run_name", "zh_study")
        path = (
            resolve_project_path(config["paths"]["runs_dir"])
            / run_name
            / "injected_recall_human_audit_sample.jsonl"
        )
    Handler.audit_path = path
    Handler.rows = list(read_jsonl(path))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(json.dumps({"url": url, "audit": str(path), "rows": len(Handler.rows)}, ensure_ascii=False))
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
