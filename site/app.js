const app=document.getElementById('app');
const templates={home:'tpl-home',new:'tpl-new',processing:'tpl-processing',workspace:'tpl-workspace',pipeline:'tpl-pipeline',docs:'tpl-docs'};
const stages=[
['01','Load Images','12 Lighthouse images'],
['02','SIFT Feature Detection','Up to 8,000 features / image'],
['03','FLANN + Lowe Matching','kNN matching + ratio 0.75'],
['04','RANSAC Verification','27 verified image pairs'],
['05','Camera Pose Estimation','10 / 12 cameras registered'],
['06','Triangulation / SfM','1,352 sparse 3D points'],
['07','Stereo Depth','5 SGBM depth maps'],
['08','Point Cloud Filtering','1,161 final points'],
['09','Poisson Meshing','5,493 vertices · 10,829 triangles']
];
let files=[],timers=[],running=false;

function route(){const k=(location.hash||'#home').slice(1);return templates[k]?k:'home'}
function go(page){location.hash=page}
function bindNav(){document.querySelectorAll('[data-nav]').forEach(el=>el.onclick=e=>{e.preventDefault();go(el.dataset.nav)})}
function active(r){document.querySelectorAll('[data-nav]').forEach(el=>el.classList.toggle('active',el.dataset.nav===r))}
function render(){
  timers.forEach(clearTimeout);timers=[];running=false;
  const r=route(),tpl=document.getElementById(templates[r]);
  app.replaceChildren(tpl.content.cloneNode(true));active(r);bindNav();
  if(r==='new')initNew();
  if(r==='processing')initProcessing();
  window.scrollTo(0,0);
}
function initNew(){
  const input=document.getElementById('fileInput'),grid=document.getElementById('fileGrid'),badge=document.getElementById('validationBadge');
  function draw(){
    grid.innerHTML='';
    if(!files.length){grid.innerHTML='<div class="empty-grid">No local files selected. The demo uses the verified 12-image Lighthouse submission.</div>';badge.textContent='DEMO DATASET';return}
    badge.textContent=files.length+' LOCAL FILES';
    files.slice(0,12).forEach(f=>{const d=document.createElement('div');d.className='file-item';d.innerHTML='<strong>'+esc(f.name)+'</strong><small>'+fmtBytes(f.size)+'</small>';grid.appendChild(d)})
  }
  input.onchange=()=>{files=[...input.files];draw()};
  document.getElementById('clearFiles').onclick=()=>{files=[];input.value='';draw()};
  document.getElementById('startRun').onclick=()=>{sessionStorage.setItem('autoplay','1');go('processing')};
  draw();
}
function initProcessing(){
  renderStages(9);renderMetrics();
  document.getElementById('replayRun').onclick=replay;
  if(sessionStorage.getItem('autoplay')==='1'){sessionStorage.removeItem('autoplay');timers.push(setTimeout(replay,250))}
}
function renderStages(done,runningIndex=-1){
  const g=document.getElementById('stageGrid');g.innerHTML='';
  stages.forEach((s,i)=>{const a=document.createElement('article');a.className='stage '+(i<done?'done':i===runningIndex?'running':'');a.innerHTML='<b>'+s[0]+'</b><h3>'+s[1]+'</h3><p>'+s[2]+'</p>';g.appendChild(a)})
}
function renderMetrics(){
  const rows=[['Input images','12'],['Verified pairs','27'],['Registered cameras','10 / 12'],['Sparse points','1,352'],['Depth maps','5'],['Final cloud','1,161'],['Mesh vertices','5,493'],['Mesh triangles','10,829']];
  document.getElementById('processingMetrics').innerHTML=rows.map(x=>'<div class="metric"><span>'+x[0]+'</span><strong>'+x[1]+'</strong></div>').join('');
}
function replay(){
  if(running)return;running=true;let i=0;const btn=document.getElementById('replayRun');btn.disabled=true;btn.textContent='Running…';
  const tick=()=>{const pct=Math.round(i/9*100);document.getElementById('progressPct').textContent=pct+'%';document.getElementById('progressBar').style.width=pct+'%';document.getElementById('progressStatus').textContent=i>=9?'RECONSTRUCTION COMPLETE':'PROCESSING';document.getElementById('progressStage').textContent=i>=9?'9 of 9 stages':'Stage '+(i+1)+' of 9';renderStages(i,i<9?i:-1);if(i<9){i++;timers.push(setTimeout(tick,350))}else{running=false;btn.disabled=false;btn.textContent='Replay Stages'}};tick();
}
function fmtBytes(n){if(n<1024)return n+' B';if(n<1048576)return(n/1024).toFixed(1)+' KB';return(n/1048576).toFixed(1)+' MB'}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
window.addEventListener('hashchange',render);bindNav();render();
