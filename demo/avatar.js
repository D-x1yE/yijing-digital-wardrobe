
const $=s=>document.querySelector(s);let current=null,state=null,key='',candidateKey='',requesting=false,sampleFiles=null;let previewMode='model',chatSignature='',historySignature='',localUrls={};
async function api(path,p){const r=await fetch('/api/avatar/'+path,p?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}:{});const d=await r.json();if(!r.ok)throw Error(d.error);return d;}
async function data(file){return {name:file.name||'sample',data:await new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(file);})};}
async function action(action,extra={}){if(!current)throw Error('先创建或选择项目');return api('action',{id:current,action,...extra});}
async function run(fn){if(requesting)return;requesting=true;$('#chat').disabled=true;$('#composerHint').textContent='正在处理你的请求…';try{$('#error').textContent='';await fn();await refresh();}catch(e){$('#error').textContent=e.message;}finally{requesting=false;$('#chat').disabled=false;$('#composerHint').textContent='Enter 发送 · Shift + Enter 换行';}}
$('#sample').onclick=()=>run(async()=>{sampleFiles=[await fetch('sample_person.png').then(r=>r.blob())];renderUploads('people',sampleFiles);});
$('#people').onchange=()=>{sampleFiles=null;renderUploads('people',Array.from($('#people').files));};$('#garment').onchange=()=>renderUploads('garment',Array.from($('#garment').files));
$('#create').onclick=()=>run(async()=>{const people=await Promise.all(Array.from(sampleFiles||$('#people').files).map(data));const s=await api('create',{people,intent:$('#text').value||'请用这些照片建立原始三维人物 V0'});current=s.id;key='';});
$('#tryon').onclick=()=>run(async()=>{const f=$('#garment').files[0];if(!f)throw Error('请上传衣服商品图');await action('tryon',{garment:await data(f),category:$('#category').value,photo_type:$('#photo_type').value});});
$('#approve').onclick=()=>run(()=>action('approve',{review:{identity:$('#identity').checked,garment:$('#details').checked,views:$('#views').checked}}));$('#reject').onclick=()=>run(()=>action('reject'));
$('#select').onclick=()=>run(()=>action('select',{version:$('#versions').value}));$('#undo').onclick=()=>run(()=>action('undo'));$('#compareButton').onclick=()=>run(()=>action('compare'));$('#animate').onclick=()=>run(()=>action('animate',{motion:$('#motion').value}));$('#chat').onclick=()=>run(async()=>{const text=$('#text').value.trim();if(!text)return;const f=$('#garment').files[0];await action('chat',{text,...(f?{garment:await data(f),photo_type:$('#photo_type').value,category:$('#category').value}:{})});$('#text').value='';});
function iframe(job,view='front'){const f=document.createElement('iframe');f.title='三维人物';f.src='gs_viewer.html?id='+encodeURIComponent(job)+'&ui=studio&view='+view;return f;}
function show(s){state=s;$('#title').textContent=s.name+' · '+(s.display?.version||s.current||'准备 V0');$('#stage').textContent=s.stage;if(s.error)$('#error').textContent=s.error;renderChat(s.messages);
 const old=$('#versions').value;$('#versions').replaceChildren();for(const v of Object.keys(s.versions))$('#versions').add(new Option(v,v));$('#versions').value=old in s.versions?old:s.current;
 const display=s.display||{},v=s.versions[display.version||s.current],newkey=JSON.stringify([s.id,v?.job,display,v?.animations]);
 if(key!==newkey){key=newkey;previewMode=display.mode==='animation'?'video':display.mode==='compare'?'compare':'model';$('#model').replaceChildren();$('#compare').replaceChildren();$('#compare').hidden=display.mode!=='compare';if(v){if(display.mode==='compare'){for(const vv of [s.versions.V0,v]){const cell=document.createElement('div');cell.className='compare-cell';const label=document.createElement('span');label.className='compare-label';label.textContent=vv.id+(vv.id==='V0'?' · 原始版本':' · 当前对照');cell.append(label,iframe(vv.job));$('#compare').append(cell);}}else $('#model').append(iframe(v.job,display.view));}
 const job=v?.animations?.[display.motion||$('#motion').value];$('#video').pause();$('#video').hidden=!job;$('#download').hidden=!job;if(job){$('#video').src='studio_results/'+job+'/animation.mp4';$('#download').href=$('#video').src;}else{$('#video').removeAttribute('src');$('#video').load();}}
 renderReferences(s,v);$('#versionBadge').textContent=v?.id||'未创建';$('#saveState').textContent=s.busy?'正在处理…':'已保存至本机';applyPreview();
 const c=s.candidate;$('#reviewPanel').hidden=!(c&&c.status==='review');const ck=c?.id+':'+c?.status;if(candidateKey!==ck){candidateKey=ck;$('#identity').checked=$('#details').checked=$('#views').checked=false;$('#images').replaceChildren();if(c?.images){const base=s.versions[c.base];for(let i=0;i<c.images.length;i++){for(const [label,path] of [['原图 '+(i+1),base.images[i]],['衣服','avatar_sessions/'+s.id+'/'+c.garment],['候选 '+(i+1),c.images[i]]]){const fig=document.createElement('figure'),cap=document.createElement('figcaption'),im=new Image();cap.textContent=label;im.src=path;im.alt=label;fig.append(cap,im);$('#images').append(fig);}}}}
 for(const id of ['tryon','approve','reject','select','undo','animate'])$('#'+id).disabled=!!s.busy;}
async function refresh(){const r=await api('state');$('#connectionText').textContent=r.capabilities.qwen_ready?'本机服务已连接':'助手正在启动';$('#projectCount').textContent=r.sessions.length;$('#cap').textContent='Qwen 权重：'+(r.capabilities.qwen_weights_ready?'已下载':'下载未完成')+'；对话服务：'+(r.capabilities.qwen_ready?'就绪':'启动中或未启动')+'；FASHN 权重：'+(r.capabilities.fashn_weights_ready?'已下载':'下载未完成');const hk=JSON.stringify(r.sessions.map(s=>[s.id,s.current,s.busy]));if(historySignature!==hk){historySignature=hk;$('#history').replaceChildren();for(const s of r.sessions){const b=document.createElement('button');b.textContent=s.name+' · '+(s.current||'生成中');b.dataset.id=s.id;b.onclick=()=>{current=s.id;key='';show(s);};$('#history').append(b);}}if(!current&&r.sessions.length)current=r.sessions[0].id;const s=r.sessions.find(x=>x.id===current);if(s)show(s);else{renderChat([]);applyPreview();}for(const b of $('#history').children)b.setAttribute('aria-current',String(b.dataset.id===current));$('#create').disabled=!!r.active;}

function assistantCopy(content){
 if(!/^调用/.test(content))return content;
 const version=content.match(/V\d+/)?.[0]||'当前版本';
 if(content.includes('compare'))return '已打开版本对比，可以对照查看原始人物与换装后的变化。';
 if(content.includes('animate'))return '已为你安排 '+version+' 的动作预览，可在「动作视频」中查看生成结果。';
 if(content.includes('tryon'))return '已安排试衣。候选图生成后，请检查效果并确认是否保存新版本。';
 if(content.includes('reconstruct'))return '正在为你建立原始三维人物，完成后将保存为 V0。';
 return content;
}
function renderChat(messages){
 const signature=JSON.stringify(messages);if(signature===chatSignature)return;chatSignature=signature;const box=$('#messages');box.replaceChildren();
 if(!messages.length){const greeting=document.createElement('div');greeting.className='chat-empty';const h=document.createElement('h3');h.textContent='今天，想尝试什么穿搭？';const p=document.createElement('p');p.textContent='上传照片建立人物，再选择喜欢的衣服。我可以帮你查看不同视角、对比版本，或生成动作。';greeting.append(h,p);box.append(greeting);}
 for(const m of messages){const row=document.createElement('div');row.className='message '+(m.role==='user'?'user':'assistant');const role=document.createElement('span');role.className='message-role';role.textContent=m.role==='user'?'你':'衣橱助手';const text=document.createElement('p');text.textContent=m.role==='user'?m.content:assistantCopy(m.content);row.append(role,text);box.append(row);}box.scrollTop=box.scrollHeight;
}
function applyPreview(){
 const viewed=state?.versions?.[state?.display?.version||state?.current];if(previewMode==='model'&&viewed&&!$('#model').firstChild)$('#model').append(iframe(viewed.job,state?.display?.view));
 for(const b of document.querySelectorAll('[data-preview]'))b.setAttribute('aria-selected',String(b.dataset.preview===previewMode));
 const hasModel=!!$('#model').firstChild,hasVideo=!!$('#video').getAttribute('src'),hasCompare=!!$('#compare').firstChild;
 $('#model').hidden=previewMode!=='model'||!hasModel;$('#emptyState').hidden=previewMode!=='model'||hasModel;
 $('#videoPanel').hidden=previewMode!=='video';$('#video').hidden=!hasVideo;$('#videoEmpty').hidden=hasVideo;
 $('#compare').hidden=previewMode!=='compare'||!hasCompare;$('#compareEmpty').hidden=previewMode!=='compare'||hasCompare;
 $('#references').hidden=previewMode!=='references';$('#download').hidden=previewMode!=='video'||!hasVideo;$('#videoDot').hidden=!hasVideo;
 if(previewMode!=='video')$('#video').pause();$('#previewHint').textContent=({model:'拖动旋转 · 滚轮缩放 · 右键平移',video:'播放动作 · 可循环观看',compare:'原始版本与当前对照',references:'当前版本使用的参考图片'})[previewMode];
}
let referenceSignature='';
function renderReferences(s,v){const paths=(v?.images||[]).map((path,i)=>['人物参考 '+(i+1),path]);const garment=v?.tryon?.garment||(s.candidate&&s.candidate.version===v?.id?s.candidate.garment:null);if(garment)paths.push(['服装参考','avatar_sessions/'+s.id+'/'+garment]);const sig=JSON.stringify(paths);if(sig===referenceSignature)return;referenceSignature=sig;$('#references').replaceChildren();for(const [label,path] of paths){const fig=document.createElement('figure'),im=new Image(),cap=document.createElement('figcaption');im.src=path;im.alt=label;cap.textContent=label;fig.append(im,cap);$('#references').append(fig);}if(!paths.length){const p=document.createElement('p');p.className='hint';p.textContent='上传照片后，参考图片会显示在这里。';$('#references').append(p);}}
function renderUploads(id,files){(localUrls[id]||[]).forEach(URL.revokeObjectURL);localUrls[id]=[];const box=$('#'+id+'Preview');box.replaceChildren();for(const f of files){const im=new Image();im.src=URL.createObjectURL(f);im.alt=f.name||'演示人物';localUrls[id].push(im.src);box.append(im);}}
for(const b of document.querySelectorAll('[data-step]'))b.onclick=()=>{for(const x of document.querySelectorAll('[data-step]'))x.setAttribute('aria-selected',String(x===b));for(const p of document.querySelectorAll('[data-step-panel]'))p.hidden=p.dataset.stepPanel!==b.dataset.step;};
for(const b of document.querySelectorAll('[data-preview]'))if(b.id!=='compareButton')b.onclick=()=>{previewMode=b.dataset.preview;applyPreview();};
const originalCompare=$('#compareButton').onclick;$('#compareButton').onclick=()=>{if(!state||Object.keys(state.versions).length<2){previewMode='compare';applyPreview();}else originalCompare();};
$('#systemToggle').onclick=()=>{const panel=$('#systemPanel');panel.hidden=!panel.hidden;$('#systemToggle').setAttribute('aria-expanded',String(!panel.hidden));};
for(const b of document.querySelectorAll('[data-prompt]'))b.onclick=()=>{$('#text').value=b.dataset.prompt;$('#text').focus();};
$('#text').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('#chat').click();}};
for(const zone of document.querySelectorAll('.dropzone')){zone.ondragover=e=>{e.preventDefault();zone.classList.add('dragging');};zone.ondragleave=()=>zone.classList.remove('dragging');zone.ondrop=e=>{e.preventDefault();zone.classList.remove('dragging');const input=$('#'+zone.dataset.input),transfer=new DataTransfer();for(const file of e.dataTransfer.files){if(!file.type.startsWith('image/'))continue;transfer.items.add(file);if(!input.multiple)break;}input.files=transfer.files;input.dispatchEvent(new Event('change'));};}
async function loop(){try{await refresh();}catch(e){$('#error').textContent=e.message;}setTimeout(loop,3000);}loop();
