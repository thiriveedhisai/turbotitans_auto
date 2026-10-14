"""Browser side of tt_dash (served as-is; data arrives over /events and /stream.mjpg)."""

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>tt_dash</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111;--pan:#1b1b1b;--fg:#ddd;--dim:#888;--line:#333}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;display:flex;gap:4px 12px;align-items:center;padding:4px 8px;background:#000;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{color:#6cf}
header label{cursor:pointer;user-select:none;white-space:nowrap}
#conn{padding:1px 6px;border-radius:3px}
.ok{background:#174}.bad{background:#811}
main{display:flex;flex-wrap:wrap;gap:4px;padding:4px;align-content:flex-start}
.p{background:var(--pan);border:1px solid var(--line);border-radius:4px;display:flex;flex-direction:column;min-width:0;overflow:hidden;
   flex:1 1 420px;height:44vh;min-height:240px}
#statep,#tagsp{flex:1 1 300px}
#tagsp{height:auto;max-height:44vh;min-height:0}
#logp{flex:1 1 100%;height:55vh}
body.solo #logp{height:calc(100vh - 60px)}
body.solo .p:not(#logp):not(#statep):not(#tagsp){height:calc(100vh - 60px)}
.p>h3{margin:0;padding:3px 6px;font-size:12px;color:var(--dim);border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.p>h3 span{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p>h3 b{color:var(--fg)}
.body{flex:1;min-height:0;position:relative;overflow:auto}
canvas{width:100%;height:100%;display:block}
#cam img{width:100%;height:100%;object-fit:contain;background:#000}
.hidden{display:none!important}
table{border-collapse:collapse;width:100%}
td,th{padding:1px 5px;border-bottom:1px solid #262626;vertical-align:top;text-align:left}
#log{font-family:ui-monospace,monospace;font-size:12px}
.lr{padding:1px 5px;border-bottom:1px solid #262626;overflow-wrap:anywhere}
.lr .t,.lr .z{color:var(--dim);margin-right:6px;white-space:nowrap}
.lr .n{color:#9ab;margin:0 6px}
.cat{font-weight:bold;padding:0 3px;border-radius:2px;font-size:11px}
.c-START{color:#5e5}.c-ZONE{color:#000;background:#5dd}.c-STEP{color:#5cc}.c-BEHAV{color:#79f}
.c-STUCK{color:#fff;background:#b33}.c-MARKER{color:#e7e}.c-MAP{color:#000;background:#fb3}.c-SEE{color:#dc5}.c-TERRAIN{color:#6c6}
.c-ODOM{color:#999}.c-RESULT{color:#fff;background:#24a}.c-CRASH{color:#fff;background:#c00}.c-LOG{color:#aaa}
.lr.WARN .m{color:#fc6}.lr.ERROR .m,.lr.FATAL .m{color:#f77}
.lr.zone{border-top:2px solid #5dd;margin-top:3px}
#cam .body{overflow:hidden}
#state{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:1px 8px;padding:4px 6px;font-family:ui-monospace,monospace;font-size:12px}
#state div{overflow-wrap:anywhere}
#state .k{color:var(--dim)}
.stale{color:#f66}.fresh{color:#6e6}
#filters{display:flex;flex-wrap:wrap;gap:4px 8px;padding:3px 6px;border-bottom:1px solid var(--line);font-size:12px}
#filters label{white-space:nowrap}
input[type=text]{background:#000;color:var(--fg);border:1px solid var(--line);padding:1px 4px}
select{background:#000;color:var(--fg);border:1px solid var(--line)}
#tags td{font-family:ui-monospace,monospace;font-size:12px}
.TRIGGER{color:#5f5;font-weight:bold}.seen{color:#ee5}.bad-tag{color:#f77}
</style></head><body>
<header>
 <b>tt_dash</b><span id="conn" class="bad">connecting</span>
 <span id="banner"></span>
 <span style="flex:1"></span>
 <label><input type="checkbox" data-p="cam">camera</label>
 <label><input type="checkbox" data-p="lidar">lidar</label>
 <label><input type="checkbox" data-p="map">map</label>
 <label><input type="checkbox" data-p="statep">state</label>
 <label><input type="checkbox" data-p="tagsp">tags</label>
 <label><input type="checkbox" data-p="logp">log</label>
 <label id="drivelab" class="hidden"><input type="checkbox" data-p="drivep">drive</label>
 <a href="/frame.jpg" target="_blank" style="color:#6cf">snapshot</a>
</header>
<main id="main">
 <div class="p hidden" id="drivep"><h3><span><b>Drive</b> arrow keys while enabled; space = stop; Esc = hand back to the stack</span></h3>
  <div class="body" style="padding:6px;display:flex;flex-direction:column;gap:6px">
   <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <button id="drivebtn" style="font-size:15px;padding:6px 14px">Take control</button>
    <label>speed <input type="range" id="dspeed" min="0.1" max="1.0" step="0.05" value="0.4"> <span id="dspeedv">0.40</span> m/s</label>
    <button id="estopbtn" style="font-size:15px;padding:6px 14px;background:#b00;color:#fff;border:0">E-STOP</button>
    <button id="releasebtn">release e-stop</button>
   </div>
   <div id="drivestat" style="font-family:ui-monospace,monospace;font-size:14px">stack in control</div>
   <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:6px">
    <b>Map</b>
    <button id="mapstart" style="font-size:14px;padding:5px 12px">Start map</button>
    <button id="mapstop" style="font-size:14px;padding:5px 12px">Stop &amp; save map</button>
    <span id="mapstat" style="font-family:ui-monospace,monospace">not mapping</span>
   </div>
   <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:6px">
    <b>Odometry check</b>
    <label>leg 1 <input type="text" id="oc1" value="5.0" size="4"> m</label>
    <label>leg 2 <input type="text" id="oc2" value="5.0" size="4"> m</label>
    <label>turn <input type="text" id="octurn" value="90" size="3">&deg;</label>
    <button id="ocstart">Start check</button>
    <button id="ocmark" style="font-size:14px;padding:5px 14px;background:#274;color:#fff;border:0">Mark</button>
    <button id="occancel">Cancel</button>
   </div>
   <div id="ocstat" style="font-family:ui-monospace,monospace;white-space:pre-wrap">not running</div>
   <div style="color:var(--dim);font-size:12px">Commands go to /teleop_cmd; the speed governor still applies collision stops, e-stop, rollover and acceleration limits (teleop_max_speed in tuning). All perception, the mission and logging keep running. Commands stop as soon as you hand back, the page loses focus, or the connection drops; the stack takes over again 0.3 s later.</div>
  </div></div>
 <div class="p" id="cam"><h3><span title="Tag boxes: green = close enough to trigger its zone, yellow = seen, red = rejected (reason shown), grey = square found but not decoded. Dashed line = camera height. White line = bright exit."><b>Camera</b> tags: green triggers, yellow seen, red rejected, grey undecoded</span></h3>
  <div class="body"><img id="camimg" alt="no camera image yet"></div></div>
 <div class="p" id="lidar"><h3><span title="Green = returns the stack uses, grey = masked as own truck, blue/orange dashed = steering arc being sent, cyan = gaps, magenta = hoop gate"><b>Lidar</b> forward = up</span>
  <select id="lrange"><option>1.5</option><option selected>3</option><option>6</option><option>12</option></select> m</h3>
  <div class="body"><canvas id="lc"></canvas></div></div>
 <div class="p" id="map"><h3><span title="Lidar map (white walls), odometry trail coloured by the zone the mission was in, white arrow = odometry, orange = map localization, green = current scan, gaps/gates"><b>Map</b> white = odometry, orange = map pose</span>
  <label><input type="checkbox" id="follow" checked>follow</label></h3>
  <div class="body"><canvas id="mc"></canvas></div></div>
 <div class="p" id="statep"><h3><span><b>State</b> what it is doing</span></h3><div class="body"><div id="state"></div></div></div>
 <div class="p" id="tagsp"><h3><span><b>Tags</b> in view (placement check)</span></h3>
  <div class="body"><table id="tags"><thead><tr><th>id</th><th>zone</th><th>dist</th><th>bearing</th><th>face</th><th>px</th><th>status</th></tr></thead><tbody></tbody></table></div></div>
 <div class="p" id="logp"><h3><span><b>Log</b> decisions, one timeline</span><span id="logcount" style="flex:0"></span></h3>
  <div id="filters"></div>
  <div class="body" id="logbody"><div id="log"></div></div></div>
</main>
<script>
const CATS=['START','ZONE','STEP','BEHAV','STUCK','MARKER','MAP','SEE','TERRAIN','ODOM','RESULT','CRASH','LOG'];
const VIEWS={all:['cam','lidar','map','statep','tagsp','logp'],cam:['cam','tagsp','statep','logp'],
             lidar:['lidar','map','statep','logp'],log:['statep','logp'],map:['map','lidar','statep','logp']};
const q=new URLSearchParams(location.search);
let cfg={}, events=[], zonesSeen=[], snap=null, trail=[], lastZoneRow=null, mapImg=null, mapInfo=null;
const $=id=>document.getElementById(id);
function el(tag,txt,cls){const e=document.createElement(tag);if(txt!==undefined)e.textContent=txt;if(cls)e.className=cls;return e;}

// ---- panels
function setPanels(list){
  document.querySelectorAll('header input[data-p]').forEach(cb=>{cb.checked=list.includes(cb.dataset.p);});
  applyPanels();
}
function applyPanels(){
  const on=[...document.querySelectorAll('header input[data-p]')].filter(c=>c.checked).map(c=>c.dataset.p);
  ['cam','lidar','map','statep','tagsp','logp','drivep'].forEach(p=>$(p).classList.toggle('hidden',!on.includes(p)));
  document.body.classList.toggle('solo',on.filter(p=>!['statep','tagsp'].includes(p)).length<=1);
  if(on.includes('cam')&&!$('camimg').src.includes('stream'))$('camimg').src='/stream.mjpg';
  if(!on.includes('cam'))$('camimg').removeAttribute('src');
}
document.querySelectorAll('header input[data-p]').forEach(cb=>cb.onchange=applyPanels);

// ---- log filters
const F={cats:new Set(CATS.filter(c=>c!=='LOG')),zone:'',text:'',chatter:false,info:false,auto:true};
function buildFilters(){
  const f=$('filters');f.textContent='';
  CATS.forEach(c=>{const l=el('label');const cb=el('input');cb.type='checkbox';cb.checked=F.cats.has(c);
    cb.onchange=()=>{cb.checked?F.cats.add(c):F.cats.delete(c);render();};l.append(cb,el('span',c,'cat c-'+c));f.append(l);});
  const zs=el('select');zs.id='zsel';zs.append(new Option('all zones',''));zs.onchange=()=>{F.zone=zs.value;render();};f.append(zs);
  const t=el('input');t.type='text';t.placeholder='search (regex)';t.oninput=()=>{F.text=t.value;render();};f.append(t);
  [['chatter','chatter'],['info','plain INFO'],['auto','auto-scroll']].forEach(([k,lab])=>{
    const l=el('label');const cb=el('input');cb.type='checkbox';cb.checked=F[k];
    cb.onchange=()=>{F[k]=cb.checked;if(k!=='auto')render();};l.append(cb,document.createTextNode(lab));f.append(l);});
}
function match(e){
  if(e.noise&&!F.chatter)return false;
  if(e.cat==='LOG'&&e.lvl==='INFO'&&!F.info)return false;
  if(!F.cats.has(e.cat)&&!(e.cat==='LOG'&&F.info&&e.lvl==='INFO'))return false;
  if(F.zone&&e.zone!==F.zone)return false;
  if(F.text){try{if(!new RegExp(F.text,'i').test(e.msg+' '+e.node))return false;}catch(_){return false;}}
  return true;
}
function row(e){
  const r=el('div');r.className='lr '+e.lvl+(e.cat==='ZONE'&&e.msg.startsWith('===')?' zone':'');
  const t=e.el!=null?e.el.toFixed(1)+'s':(e.t?new Date(e.t*1000).toLocaleTimeString():'-');
  r.append(el('span',t,'t'),el('span',e.zone+'/'+e.step,'z'),el('span',e.cat,'cat c-'+e.cat),
           el('span',e.node,'n'),el('span',e.msg,'m'));return r;
}
const MAXROWS=4000;
function render(){
  const tb=$('log');tb.textContent='';
  const sel=events.filter(match).slice(-MAXROWS);sel.forEach(e=>tb.append(row(e)));
  $('logcount').textContent=sel.length+'/'+events.length;scrollLog();
}
function scrollLog(){if(F.auto){const b=$('logbody');b.scrollTop=b.scrollHeight;}}
function addEvents(items){
  const tb=$('log');let added=0;
  for(const e of items){events.push(e);
    if(e.zone&&e.zone!=='-'&&!zonesSeen.includes(e.zone)){zonesSeen.push(e.zone);$('zsel').append(new Option(e.zone,e.zone));}
    if(match(e)){tb.append(row(e));added++;}}
  while(tb.childElementCount>MAXROWS)tb.firstElementChild.remove();
  if(added){$('logcount').textContent=tb.childElementCount+'/'+events.length;scrollLog();}
}

// ---- state panel
function fmt(v){if(v==null)return '-';if(typeof v==='number')return Number.isInteger(v)?String(v):(Math.abs(v)>=100?v.toFixed(0):v.toFixed(2));
  if(typeof v==='object')return JSON.stringify(v);return String(v);}
function showState(s){
  const st=s.state||{}, rows=[];
  const ms=st.mission||{}, bs=st.bstatus||{}, cmd=st.bcmd||{};
  const deg=r=>r==null?'-':(r*180/Math.PI).toFixed(0)+'\u00b0';
  rows.push(['race',(ms.started?(ms.finished?'FINISHED':'running'):'waiting for start')+'  t='+fmt(ms.elapsed)+'s'+(st.estop?'  E-STOP':'')]);
  const hw=st.hw_estop;
  rows.push(['hw e-stop',hw?((hw.stop?'STOP':'RUN')+'  '+hw.reason+'  hb '+hw.heartbeat+' pin '+hw.pin_state+'  age '+hw.age+'s'):'not running']);
  rows.push(['zone / step',(ms.zone||'-')+' / '+fmt(ms.step)+(ms.phase?'  phase '+ms.phase:'')]);
  rows.push(['maneuver',(bs.behavior||'-')+'  '+(bs.status||'')+(bs.info?'  ('+bs.info+')':'')]);
  rows.push(['params',cmd.params?JSON.stringify(cmd.params):'-']);
  const raw=st.cmd_raw, out=st.cmd;
  rows.push(['wants v/steer',raw?raw[0].toFixed(2)+' m/s  '+deg(raw[1]):'-']);
  rows.push(['sent v/steer',out?out[0].toFixed(2)+' m/s  '+deg(out[1])+((raw&&Math.abs(raw[0])-Math.abs(out[0])>0.05)?'  (governor limiting)':''):'-']);
  rows.push(['speed caps','mission '+fmt(st.cap_mission)+'  terrain '+fmt(st.cap_terrain)]);
  const at=st.attitude;
  rows.push(['terrain',(st.terrain||'-')+(at?'  roll '+at[0].toFixed(1)+'\u00b0 nose-up '+at[1].toFixed(1)+'\u00b0 vib '+at[2].toFixed(2):'')]);
  rows.push(['lidar',st.lidar_ok===false?'GATED (tilted)':'ok']);
  const p=s.pose;rows.push(['odom',p?'x '+p[0].toFixed(2)+' y '+p[1].toFixed(2)+' yaw '+deg(p[2])+(st.odom_degraded?'  DEGRADED (dead-reckoning)':''):'-']);
  rows.push(['bright exit',st.bright==null||isNaN(st.bright)?'-':deg(st.bright)+' (+ = left)']);
  const tg=st.detections||[];
  rows.push(['tags (detector)',tg.length?tg.map(t=>t.zone+' '+t.distance+'m '+t.bearing_deg+'\u00b0').join(', '):'-']);
  if(st.tag_source)rows.push(['camera boxes from',st.tag_source]);
  const lc=st.loc;if(lc)rows.push(['map localization',lc.state+'  quality '+lc.quality+'  zone '+(lc.zone||'-')+'  ('+lc.ms+' ms)']);
  const box=$('state');box.textContent='';
  rows.forEach(([k,v])=>{box.append(el('div',k,'k'),el('div',v));});
  box.append(el('div','topics','k'));const tp=el('div');
  Object.entries(s.topics||{}).forEach(([n,t])=>{const sp=el('span',n+' '+t.hz.toFixed(1)+'Hz  ',t.age>1.5?'stale':'fresh');tp.append(sp);});
  box.append(tp);
  $('banner').textContent=(ms.zone?'ZONE '+ms.zone+'/'+ms.step+'  ':'')+(bs.behavior?bs.behavior+' '+(bs.status||''):'');
  const mp=st.mapping;
  if(mp&&$('mapstat'))$('mapstat').textContent=(mp.mapping?'MAPPING  ':'done  ')+mp.path_m+' m, '+mp.scans+' scans'+
    (mp.quality!=null?', match '+mp.quality:'')+(mp.loop?'  loop closed ('+mp.loop[0]+' m, '+mp.loop[1]+'\u00b0)':'')+
    (mp.saved?'  saved '+mp.saved:'  -> '+mp.dir);
  const oc=st.odocheck;
  if(oc&&$('ocstat')){
    let t=oc.active?'RUNNING (teleop capped at '+oc.max_speed+' m/s)  mark '+oc.marks+'/4: '+oc.next:'not running';
    const r=oc.result;
    if(r)t='odometry measured '+r.leg1_odom_m+' / '+r.leg2_odom_m+' m  ->  scale '+r.scale+
      (r.suggested_odom_scale?'   set odom_guard odom_scale: '+r.suggested_odom_scale:'')+(r.warning?'   !! '+r.warning:'')+
      '\nturn error vs '+r.true_turn_deg+'\u00b0:  path '+r.path_turn_err_deg+'\u00b0  ZED '+r.zed_turn_err_deg+
      '\u00b0  gyro '+r.gyro_turn_err_deg+'\u00b0     heading drift along legs: ZED '+r.zed_drift_leg1_deg+
      ' / '+r.zed_drift_leg2_deg+'\u00b0, gyro '+r.gyro_drift_leg1_deg+' / '+r.gyro_drift_leg2_deg+'\u00b0\nsaved '+r.saved;
    if(oc.error)t+='  ('+oc.error+')';
    $('ocstat').textContent=t;
  }
}
function showTags(tags){
  const tb=$('tags').tBodies[0];tb.textContent='';
  (tags||[]).forEach(t=>{const tr=el('tr');
    [t.id,t.zone||'?',t.dist==null?'-':t.dist.toFixed(2)+' m',t.bearing==null?'-':t.bearing.toFixed(0)+'\u00b0',
     t.normal==null?'-':t.normal.toFixed(0)+'\u00b0',t.side_px.toFixed(0)].forEach(v=>tr.append(el('td',String(v))));
    tr.append(el('td',t.status,t.status==='TRIGGER'?'TRIGGER':(t.status==='seen'?'seen':'bad-tag')));tb.append(tr);});
}

// ---- lidar view
function fit(c){const r=c.getBoundingClientRect();const d=devicePixelRatio||1;
  if(c.width!==Math.round(r.width*d)||c.height!==Math.round(r.height*d)){c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);}
  return [c.getContext('2d'),c.width,c.height,d];}
function drawLidar(s){
  const [g,W,H,d]=fit($('lc'));g.clearRect(0,0,W,H);
  const R=parseFloat($('lrange').value), k=Math.min(W,H)/(2*R)*0.95, ox=W/2, oy=H/2;
  const P=(x,y)=>[ox-y*k,oy-x*k];  // x forward = up, y left = left
  g.strokeStyle='#2a2a2a';g.lineWidth=1*d;g.font=(10*d)+'px sans-serif';g.fillStyle='#555';
  for(let r=0.5;r<=R+1e-6;r+=(R>4?1:0.5)){g.beginPath();g.arc(ox,oy,r*k,0,2*Math.PI);g.stroke();g.fillText(r+'m',ox+3,oy-r*k+10*d);}
  g.beginPath();g.moveTo(ox,0);g.lineTo(ox,H);g.moveTo(0,oy);g.lineTo(W,oy);g.stroke();
  const v=cfg.vehicle||{};const f=v.lidar_to_front||0.15, rr=v.lidar_to_rear||0.28, w=(v.width||0.2)/2;
  g.strokeStyle='#6cf';g.lineWidth=2*d;g.beginPath();
  [[f,w],[f,-w],[-rr,-w],[-rr,w],[f,w]].forEach(([x,y],i)=>{const [a,b]=P(x,y);i?g.lineTo(a,b):g.moveTo(a,b);});g.stroke();
  const st=(s&&s.state)||{};
  // steering arc of the command actually sent, from the rear axle
  const cmd=st.cmd;if(cmd){const wb=v.wheelbase||0.32, ax=-(v.lidar_ahead_of_axle||0.23), kap=Math.tan(cmd[1])/wb, dir=cmd[0]<0?-1:1;
    g.strokeStyle=cmd[0]<0?'#f84':'#4af';g.setLineDash([6*d,4*d]);g.beginPath();
    for(let i=0;i<=40;i++){const sl=dir*1.5*i/40;let x,y;if(Math.abs(kap)<1e-4){x=sl;y=0;}else{x=Math.sin(kap*sl)/kap;y=(1-Math.cos(kap*sl))/kap;}
      const [a,b]=P(ax+x,y);i?g.lineTo(a,b):g.moveTo(a,b);}g.stroke();g.setLineDash([]);}
  const sc=s&&s.scan;
  if(sc){g.fillStyle='#666';sc.masked.forEach(([x,y])=>{const [a,b]=P(x,y);g.fillRect(a-2*d,b-2*d,4*d,4*d);});
    g.fillStyle=st.lidar_ok===false?'#a55':'#4f4';sc.pts.forEach(([x,y])=>{const [a,b]=P(x,y);g.fillRect(a-1.5*d,b-1.5*d,3*d,3*d);});}
  (s&&s.percep_local||[]).filter(p=>p.age<5).forEach(p=>{const [a,b]=P(p.x,p.y);const col=p.name==='gate'?'#f5f':'#5ff';
    g.globalAlpha=p.age<1?1:0.4;
    g.strokeStyle=col;g.fillStyle=col;g.lineWidth=2*d;g.beginPath();g.arc(a,b,6*d,0,2*Math.PI);g.stroke();
    const [a2,b2]=P(p.x+0.4*Math.cos(p.yaw),p.y+0.4*Math.sin(p.yaw));g.beginPath();g.moveTo(a,b);g.lineTo(a2,b2);g.stroke();
    g.fillText(p.name+' '+p.age.toFixed(1)+'s',a+8*d,b);g.globalAlpha=1;});
  if(st.lidar_ok===false){g.fillStyle='#f55';g.font='bold '+(14*d)+'px sans-serif';g.fillText('LIDAR GATED (tilted): behaviors ignore it',10*d,20*d);}
  if(!sc){g.fillStyle='#888';g.font=(14*d)+'px sans-serif';g.fillText('no /scan yet',10*d,20*d);}
}

// ---- map view
const ZC={};function zcol(z){if(!ZC[z]){let h=0;for(const ch of z)h=(h*31+ch.charCodeAt(0))%360;ZC[z]='hsl('+h+',70%,55%)';}return ZC[z];}
function drawMap(s){
  const [g,W,H,d]=fit($('mc'));g.clearRect(0,0,W,H);
  const pose=s&&s.pose;const pts=trail.map(t=>[t[0],t[1]]);
  let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);if(pose){xs.push(pose[0]);ys.push(pose[1]);}
  if(mapInfo){xs.push(mapInfo.origin[0],mapInfo.origin[0]+mapInfo.w*mapInfo.res);ys.push(mapInfo.origin[1],mapInfo.origin[1]+mapInfo.h*mapInfo.res);}
  if(!xs.length){g.fillStyle='#888';g.font=(14*d)+'px sans-serif';g.fillText('no odometry yet',10*d,20*d);return;}
  let cx,cy,span;
  const lp=s&&s.loc_pose, me=lp||pose;
  if($('follow').checked&&me){cx=me[0];cy=me[1];span=8;}
  else{const x0=Math.min(...xs)-1,x1=Math.max(...xs)+1,y0=Math.min(...ys)-1,y1=Math.max(...ys)+1;cx=(x0+x1)/2;cy=(y0+y1)/2;span=Math.max(x1-x0,y1-y0,4);}
  const k=Math.min(W,H)/span;const P=(x,y)=>[W/2+(x-cx)*k,H/2-(y-cy)*k];
  if(mapImg&&mapImg.complete){const [a,b]=P(mapInfo.origin[0],mapInfo.origin[1]+mapInfo.h*mapInfo.res);
    g.imageSmoothingEnabled=false;g.drawImage(mapImg,a,b,mapInfo.w*mapInfo.res*k,mapInfo.h*mapInfo.res*k);}
  g.strokeStyle='#222';g.lineWidth=1;const gs=1;
  for(let x=Math.floor((cx-span)/gs)*gs;x<cx+span;x+=gs){const [a]=P(x,0);g.beginPath();g.moveTo(a,0);g.lineTo(a,H);g.stroke();}
  for(let y=Math.floor((cy-span)/gs)*gs;y<cy+span;y+=gs){const [,b]=P(0,y);g.beginPath();g.moveTo(0,b);g.lineTo(W,b);g.stroke();}
  g.lineWidth=3*d;let prevZ=null;g.font=(11*d)+'px sans-serif';
  for(let i=1;i<trail.length;i++){const [x0,y0]=trail[i-1],[x1,y1,z]=trail[i];g.strokeStyle=zcol(z||'-');
    const [a0,b0]=P(x0,y0),[a1,b1]=P(x1,y1);g.beginPath();g.moveTo(a0,b0);g.lineTo(a1,b1);g.stroke();
    if(z!==prevZ){g.fillStyle=zcol(z||'-');g.fillText(z||'-',a1+5*d,b1-5*d);prevZ=z;}}
  if(me&&s.scan){const off=cfg.laser_offset||[-0.15,0];const c=Math.cos(me[2]),sn=Math.sin(me[2]);
    const lx=me[0]+c*off[0]-sn*off[1], ly=me[1]+sn*off[0]+c*off[1];g.fillStyle='#4f4';
    s.scan.pts.forEach(([x,y])=>{const [a,b]=P(lx+c*x-sn*y,ly+sn*x+c*y);g.fillRect(a-1*d,b-1*d,2*d,2*d);});}
  Object.entries((s&&s.percep)||{}).filter(([n,p])=>Date.now()/1000-p[3]<5).forEach(([n,p])=>{const [a,b]=P(p[0],p[1]);g.strokeStyle=n==='gate'?'#f5f':'#5ff';
    g.lineWidth=2*d;g.beginPath();g.arc(a,b,5*d,0,2*Math.PI);g.stroke();});
  const arrow=(p,col)=>{g.fillStyle=col;g.beginPath();
    const c=Math.cos(p[2]),sn=Math.sin(p[2]);const tip=P(p[0]+0.35*c,p[1]+0.35*sn);
    const l=P(p[0]-0.1*c-0.15*sn,p[1]-0.1*sn+0.15*c),r=P(p[0]-0.1*c+0.15*sn,p[1]-0.1*sn-0.15*c);
    g.moveTo(...tip);g.lineTo(...l);g.lineTo(...r);g.closePath();g.fill();};
  if(pose)arrow(pose,'#fff');
  if(lp)arrow(lp,'#f90');
}

// ---- drive (only with tt_dash --control)
const D={on:false,keys:{},steer:0,timer:null,last:0};
function post(path,obj){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-TT-Token':document.body.dataset.token},body:JSON.stringify(obj)}).catch(()=>{});}
function driveTick(){
  const maxS=(cfg.vehicle&&cfg.vehicle.max_steer_cmd)||0.5, sp=parseFloat($('dspeed').value);
  const now=performance.now(), dt=Math.min(0.2,(now-D.last)/1000);D.last=now;
  const want=D.keys.ArrowLeft?maxS:(D.keys.ArrowRight?-maxS:0);
  const rate=2.5*dt;D.steer+=Math.max(-rate,Math.min(rate,want-D.steer));
  const v=D.keys.ArrowUp?sp:(D.keys.ArrowDown?-Math.min(sp,0.5):0);
  post('/teleop',{v:v,steer:D.steer});
  $('drivestat').textContent='MANUAL  v='+v.toFixed(2)+' m/s  steer='+(D.steer*180/Math.PI).toFixed(0)+'\u00b0';
}
function setDrive(on){
  D.on=on;D.keys={};D.steer=0;clearInterval(D.timer);
  $('drivebtn').textContent=on?'Hand back to stack (Esc)':'Take control';
  $('drivebtn').style.background=on?'#c80':'';
  if(on){D.last=performance.now();D.timer=setInterval(driveTick,100);}
  else $('drivestat').textContent='stack in control';
}
function initDrive(){
  if(!cfg.control||D.init)return;
  D.init=true;
  $('drivelab').classList.remove('hidden');
  document.querySelector('input[data-p="drivep"]').checked=true;applyPanels();
  $('drivebtn').onclick=()=>setDrive(!D.on);
  $('dspeed').oninput=()=>{$('dspeedv').textContent=parseFloat($('dspeed').value).toFixed(2);};
  $('estopbtn').onclick=()=>{setDrive(false);post('/estop',{on:true});};
  $('releasebtn').onclick=()=>post('/estop',{on:false});
  $('mapstart').onclick=()=>post('/map/start',{});
  $('mapstop').onclick=()=>post('/map/stop',{});
  $('ocstart').onclick=()=>post('/odocheck',{action:'start',leg1_m:parseFloat($('oc1').value),
    leg2_m:parseFloat($('oc2').value),turn_deg:parseFloat($('octurn').value)});
  $('ocmark').onclick=()=>post('/odocheck',{action:'mark'});
  $('occancel').onclick=()=>post('/odocheck',{action:'cancel'});
  addEventListener('keydown',e=>{
    if(!D.on)return;
    if(e.key==='Escape'){setDrive(false);return;}
    if(e.key===' '){D.keys={};e.preventDefault();return;}
    if(e.key.startsWith('Arrow')){D.keys[e.key]=true;e.preventDefault();}
  });
  addEventListener('keyup',e=>{if(e.key.startsWith('Arrow'))D.keys[e.key]=false;});
  // Lost key-up events would leave the truck driving: drop control instead.
  addEventListener('blur',()=>{if(D.on)setDrive(false);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&D.on)setDrive(false);});
}

// ---- stream
function connect(){
  const es=new EventSource('/events');
  es.onopen=()=>{$('conn').textContent='live';$('conn').className='ok';};
  es.onerror=()=>{$('conn').textContent='disconnected (retrying)';$('conn').className='bad';if(D.on)setDrive(false);};
  es.onmessage=ev=>{const m=JSON.parse(ev.data);
    if(m.type==='hello'){cfg=m.cfg;events=[];trail=[];zonesSeen=[];buildFilters();render();initDrive();
      if(cfg.offline){$('conn').textContent='offline log';}}
    else if(m.type==='events')addEvents(m.items);
    else if(m.type==='map'){mapInfo=m;mapImg=new Image();mapImg.onload=()=>drawMap(snap);mapImg.src='data:image/png;base64,'+m.png;}
    else if(m.type==='snap'){snap=m;if(m.trail)trail.push(...m.trail);showState(m);showTags(m.tags);
      if(!$('lidar').classList.contains('hidden'))drawLidar(m);if(!$('map').classList.contains('hidden'))drawMap(m);}
  };
}
$('lrange').onchange=()=>drawLidar(snap);
setPanels(VIEWS[q.get('view')||document.body.dataset.view||'all']||VIEWS.all);
buildFilters();connect();
</script></body></html>
'''
