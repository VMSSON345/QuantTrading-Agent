
const API = "";
const PAGES = {
  mining:  {el:"page-mining",  title:"Alpha Mining"},
  kb:      {el:"page-kb",      title:"Knowledge Base"},
  predict: {el:"page-predict", title:"Dự báo cổ phiếu"},
  tracker: {el:"page-tracker", title:"Self-Improvement"},
};

function navigate(page) {
  document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.remove("active"));
  document.getElementById(PAGES[page].el).classList.add("active");
  document.querySelector(`[data-page="${page}"]`).classList.add("active");
  document.getElementById("page-title").textContent = PAGES[page].title;
  if(page==="kb")      loadKB();
  if(page==="predict") loadSymbols();
  if(page==="tracker") loadTracker();
}
document.querySelectorAll(".nav-item").forEach(a=>{
  a.addEventListener("click",e=>{e.preventDefault();navigate(a.dataset.page);});
});

// Theme
let theme="dark";
document.querySelector("[data-theme-toggle]")?.addEventListener("click",()=>{
  theme=theme==="dark"?"light":"dark";
  document.documentElement.setAttribute("data-theme",theme);
});

// Health check
async function checkAPI(){
  try{
    const r=await fetch(`${API}/api/health`);
    const ok=r.ok;
    document.getElementById("api-status").className="status-dot "+(ok?"ok":"err");
    document.getElementById("api-status-text").textContent=ok?"API sẵn sàng":"API lỗi";
  }catch{
    document.getElementById("api-status").className="status-dot err";
    document.getElementById("api-status-text").textContent="Mất kết nối";
  }
}
checkAPI(); setInterval(checkAPI,30000);

// Log
function log(msg,type="info"){
  const p=document.getElementById("mining-log");
  const d=document.createElement("div");
  d.className=`log-${type}`;
  d.textContent=`[${new Date().toLocaleTimeString("vi-VN")}] ${msg}`;
  p.appendChild(d); p.scrollTop=p.scrollHeight;
}

// Mining
async function startMining(){
  const idea=document.getElementById("idea-input").value.trim();
  if(!idea){alert("Vui lòng nhập ý tưởng giao dịch");return;}
  const btn=document.getElementById("mine-btn");
  btn.disabled=true; btn.textContent="⏳ Đang mining...";
  document.getElementById("mining-log").innerHTML="";
  document.getElementById("result-card").classList.add("hidden");
  log("Khởi động Inner Loop — Writer Agent đang viết code alpha...","info");
  try{
    const body={
      idea,
      max_inner_iter:parseInt(document.getElementById("max-inner").value),
      max_outer_iter:parseInt(document.getElementById("max-outer").value),
    };
    const res=await fetch(`${API}/api/alpha/mine`,{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body),
    });
    const data=await res.json();
    if(data.success){
      (data.outer_history||[]).forEach(h=>{
        log(`Outer iter ${h.iter}: IC=${(h.ic??0).toFixed(4)} | ${(h.review||"").slice(0,80)}...`,"info");
      });
      log("✅ Hoàn tất"+(data.added_to_kb?" — đã lưu vào Knowledge Base":""),"success");
      showResult(data);
    }else{
      log("❌ Lỗi: "+(data.error||"Unknown"),"error");
    }
  }catch(e){log("❌ Lỗi kết nối: "+e.message,"error");}
  finally{btn.disabled=false;btn.textContent="⚡ Bắt đầu Mining";}
}

function showResult(data){
  // Hiện cả 2 card
  document.getElementById("result-card").classList.remove("hidden");
  document.getElementById("interp-card").classList.remove("hidden");
}

// KB
async function loadKB(){
  const minIC=parseFloat(document.getElementById("filter-ic").value||"0");
  try{
    const [lr,sr]=await Promise.all([
      fetch(`${API}/api/kb/list?min_ic=${minIC}&top_k=100`),
      fetch(`${API}/api/kb/stats`),
    ]);
    const list=await lr.json();
    const stats=await sr.json();
    document.getElementById("kb-stats").innerHTML=[
      ["Tổng alphas",stats.total],
      ["Avg IC",(stats.avg_ic??0).toFixed(4)],
      ["Avg Sharpe",(stats.avg_sharpe??0).toFixed(4)],
      ["Best IC",(stats.best_ic??0).toFixed(4)],
    ].map(([l,v])=>`<div class="stat-badge">${l}: <span>${v}</span></div>`).join("");
    const tbody=document.getElementById("kb-tbody");
    if(!list.alphas||!list.alphas.length){
      tbody.innerHTML=`<tr><td colspan="8" class="empty-cell">📭 Chưa có alpha trong KB — hãy thử mining!</td></tr>`;
      return;
    }
    tbody.innerHTML=list.alphas.map(a=>`<tr>
      <td style="font-family:monospace;color:var(--primary)">${a.alpha_id}</td>
      <td>${a.name}</td>
      <td class="${a.metrics.ic>0?'pos':'neg'}">${a.metrics.ic.toFixed(4)}</td>
      <td>${a.metrics.sharpe.toFixed(4)}</td>
      <td style="color:var(--error)">${a.metrics.max_drawdown.toFixed(4)}</td>
      <td>${(a.metrics.win_rate*100).toFixed(1)}%</td>
      <td>${a.judge_score.toFixed(2)}</td>
      <td><span class="a-link" onclick="showCode('${a.alpha_id}',this.dataset.code)" data-code="${encodeURIComponent(a.code)}">Xem code</span></td>
    </tr>`).join("");
  }catch(e){
    document.getElementById("kb-tbody").innerHTML=`<tr><td colspan="8" class="empty-cell">Lỗi: ${e.message}</td></tr>`;
  }
}

// Predict
async function loadSymbols(){
  try{
    const r=await fetch(`${API}/api/alpha/symbols`);
    const data=await r.json();
    const sel=document.getElementById("predict-symbol");
    sel.innerHTML=(data.symbols||[]).map(s=>`<option value="${s}">${s}</option>`).join("");
    if(!data.symbols?.length) sel.innerHTML=`<option>Chưa có dữ liệu CSV</option>`;
  }catch{}
}

async function runPredict(){
  const symbol=document.getElementById("predict-symbol").value;
  const retrain=document.getElementById("retrain-cb").checked;
  const area=document.getElementById("predict-result");
  area.innerHTML=`<div class="log-placeholder">⏳ Đang dự báo ${symbol}...</div>`;
  try{
    const r=await fetch(`${API}/api/predict/run`,{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({symbol,retrain}),
    });
    const data=await r.json();
    if(data.error&&!data.predicted_return_5d){
      area.innerHTML=`<div style="color:var(--error)">❌ ${data.error}</div>`;return;
    }
    const ret=data.predicted_return_5d;
    const color=ret>=0?"var(--success)":"var(--error)";
    area.innerHTML=`<div class="metric-card">
      <div class="metric-label">Dự báo return 5 ngày — ${symbol}</div>
      <div class="metric-value" style="color:${color};font-size:36px">${ret>=0?"+":""}${(ret*100).toFixed(2)}%</div>
    </div>`;
    if(data.top_features?.length){
      Plotly.newPlot("feature-chart",[{
        type:"bar",orientation:"h",
        x:data.top_features.map(f=>f.importance),
        y:data.top_features.map(f=>f.name),
        marker:{color:"rgba(79,152,163,0.85)"},
      }],{
        paper_bgcolor:"transparent",plot_bgcolor:"transparent",
        font:{color:"#cdccca",size:12},
        margin:{l:150,r:20,t:10,b:40},
        xaxis:{title:"Importance"},showlegend:false,
      },{responsive:true,displayModeBar:false});
    }
  }catch(e){area.innerHTML=`<div style="color:var(--error)">❌ ${e.message}</div>`;}
}

// Tracker
async function loadTracker(){
  try{
    const [sr,lr]=await Promise.all([
      fetch(`${API}/api/kb/stats`),
      fetch(`${API}/api/kb/list?top_k=200`),
    ]);
    const stats=await sr.json();
    const list=await lr.json();
    const alphas=(list.alphas||[]).sort((a,b)=>a.created_at.localeCompare(b.created_at));
    document.getElementById("tracker-stats").innerHTML=[
      ["Total Alpha",stats.total,"neu"],
      ["Avg IC",(stats.avg_ic??0).toFixed(4),"pos"],
      ["Avg Sharpe",(stats.avg_sharpe??0).toFixed(4),"pos"],
      ["Best IC",(stats.best_ic??0).toFixed(4),"pos"],
    ].map(([l,v,c])=>`<div class="metric-card"><div class="metric-label">${l}</div>
      <div class="metric-value ${c}">${v}</div></div>`).join("");
    if(alphas.length){
      const xs=alphas.map((_,i)=>i+1);
      Plotly.newPlot("ic-chart",[{
        x:xs,y:alphas.map(a=>a.metrics.ic),type:"scatter",mode:"lines+markers",
        line:{color:"#4f98a3"},fill:"tozeroy",fillcolor:"rgba(79,152,163,0.1)",
      }],{paper_bgcolor:"transparent",plot_bgcolor:"transparent",
        font:{color:"#cdccca",size:12},margin:{l:50,r:20,t:10,b:40},
        xaxis:{title:"Alpha #"},yaxis:{title:"IC"},showlegend:false},
        {responsive:true,displayModeBar:false});
      Plotly.newPlot("kb-growth-chart",[{
        x:xs,y:xs,type:"scatter",mode:"lines",
        line:{color:"#6daa45"},fill:"tozeroy",fillcolor:"rgba(109,170,69,0.1)",
      }],{paper_bgcolor:"transparent",plot_bgcolor:"transparent",
        font:{color:"#cdccca",size:12},margin:{l:50,r:20,t:10,b:40},
        xaxis:{title:"Thứ tự"},yaxis:{title:"Số Alpha"},showlegend:false},
        {responsive:true,displayModeBar:false});
    }
  }catch{}
}

// Modal
function showCode(id,encodedCode){
  document.getElementById("modal-title").textContent=`Code Alpha — ${id}`;
  document.getElementById("modal-code").textContent=decodeURIComponent(encodedCode);
  document.getElementById("code-modal").classList.remove("hidden");
}
function closeModal(){document.getElementById("code-modal").classList.add("hidden");}
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeModal();});
