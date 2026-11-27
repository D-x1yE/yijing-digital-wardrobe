from pathlib import Path
root=Path(__file__).resolve().parent
p=root/'studio.html';s=p.read_text(encoding='utf8')
s=s.replace('viewKey=null,pendingPhoto=false;','viewKey=null,pendingPhoto=false,jobsSignature="",downloadsSignature="",recordSignature="";')
s=s.replace("const changed=current!==id;current=id;", "const changed=current!==id;const signature=JSON.stringify(m);if(!changed&&recordSignature===signature)return;recordSignature=signature;current=id;")
s=s.replace("$('#downloads').replaceChildren();for(const f of m.files||[])", "const downloadKey=m.id+JSON.stringify(m.files||[]);if(downloadsSignature!==downloadKey){downloadsSignature=downloadKey;$('#downloads').replaceChildren();for(const f of m.files||[])")
s=s.replace("$('#downloads').append(a)}if(!pendingPhoto", "$('#downloads').append(a)}}if(!pendingPhoto")
s=s.replace("records=r.jobs;$('#jobs').replaceChildren();", "records=r.jobs;const signature=JSON.stringify(records.map(m=>[m.id,m.status,m.name]));if(signature!==jobsSignature){jobsSignature=signature;$('#jobs').replaceChildren();")
s=s.replace("$('#jobs').append(b)}$('#generate')", "$('#jobs').append(b)}}$('#generate')")
p.write_text(s,encoding='utf8')
p=root/'gs_viewer.html';s=p.read_text(encoding='utf8')
s=s.replace('antialiased:false});','antialiased:false,renderMode:GS.RenderMode.OnChange});')
p.write_text(s,encoding='utf8')
