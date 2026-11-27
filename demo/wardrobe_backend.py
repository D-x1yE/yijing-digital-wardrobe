"""Wardrobe project workflow; reuse the verified LHM++ worker without faking try-on."""
from pathlib import Path
from PIL import Image, ImageOps
from urllib.parse import urlsplit
import base64, hashlib, io, json, shutil, subprocess, threading, time, uuid

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / 'wardrobe_projects'
CONFIG = ROOT / 'tryon_provider.json'
MODES = {'reconstruct', 'import_tryon', 'auto_tryon'}

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)

def _provider():
    if not CONFIG.exists():
        return None
    value = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    command = value.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('tryon_provider.json 的 command 必须为参数数组')
    if value.get('verified') is not True:
        return None
    return value

def _save_image(value, path):
    if not isinstance(value, dict) or not isinstance(value.get('data'), str):
        raise ValueError('图片输入格式无效')
    encoded = value['data'].split(',', 1)[-1]
    if len(encoded) > 28 * 1024**2:
        raise ValueError('单张图片须小于 20MB')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 20 * 1024**2:
            raise ValueError('单张图片须小于 20MB')
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('只接受 JPG、PNG 或 WebP 图片')
            if image.width * image.height > 30_000_000:
                raise ValueError('图片不能超过 3000 万像素')
            if min(image.size) < 64:
                raise ValueError('图片的宽和高均须至少 64 像素')
            image.load()
            image = ImageOps.exif_transpose(image).convert('RGBA' if 'A' in image.getbands() else 'RGB')
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            size = list(image.size)
    except (ValueError, Image.DecompressionBombError):
        raise
    except Exception as exc:
        raise ValueError('无法读取上传的图片') from exc
    return {'name': str(value.get('name', 'image'))[:120], 'path': str(path.relative_to(ROOT)).replace(chr(92), '/'),
            'size': size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def _update(project_dir, project, stage, status='running', **extra):
    project.update(stage=stage, status=status, updated=time.time(), **extra)
    project.setdefault('events', []).append({'stage': stage, 'time': time.time()})
    _write(project_dir / 'project.json', project)

def _worker(studio, directory, project, provider):
    ident = project['id']
    output = studio.STORE / ident
    model_started = False
    try:
        output.mkdir(exist_ok=True)
        source = directory / ('edited' if project['mode'] == 'import_tryon' else 'people')
        if project['mode'] == 'auto_tryon':
            _update(directory, project, '二维试衣模型推理')
            source = directory / 'edited'
            source.mkdir(exist_ok=True)
            variables = {'person_dir': str((directory/'people').resolve()), 'garment_path': str((directory/'garment.png').resolve()),
                         'output_dir': str(source.resolve()), 'prompt_path': str((directory/'prompt.txt').resolve()),
                         'category': project['category'], 'seed': str(project['seed'])}
            for key, path in list(variables.items()):
                if key.endswith(('dir', 'path')) and len(path)>2 and path[1]==':':
                    variables[key+'_linux'] = '/mnt/'+path[0].lower()+path[2:].replace(chr(92),'/')
            command = []
            for token in provider['command']:
                for key, value in variables.items():
                    token = token.replace('{'+key+'}', value)
                command.append(token)
            with (directory/'tryon.log').open('w', encoding='utf8') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=min(max(int(provider.get('timeout_seconds', 1800)), 60), 7200), cwd=str(ROOT.parent))
            expected = [source/f'{i:02d}.png' for i in range(len(project['people']))]
            if any(not p.is_file() for p in expected):
                raise RuntimeError('试衣提供器未输出与人物视图一一对应的 00.png、01.png 等文件')
            for path in expected:
                with Image.open(path) as image:
                    image.verify()
            project['edited'] = [{'path': str(p.relative_to(ROOT)).replace(chr(92),'/'), 'name': p.name,
                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in expected]
        _update(directory, project, 'LHM++ 预处理与三维联合重建')
        uploads = output/'uploads'
        uploads.mkdir(exist_ok=True)
        for i in range(len(project['people'])):
            shutil.copy2(source/f'{i:02d}.png', uploads/f'{i:02d}.png')
        shutil.copy2(uploads/'00.png', output/'upload.png')
        record = {'id': ident, 'name': project['name'], 'kind': 'gaussian', 'status': 'running',
                  'created': project['created'], 'source': 'LHM++；试衣来源见关联项目记录', 'files': [], 'wardrobe_project': ident}
        studio.save(output, record)
        model_started = True
        studio.run(output, record)
        if record['status'] != 'complete':
            raise RuntimeError(record.get('error', '三维重建失败'))
        _update(directory, project, '完成', 'complete', metrics=record['metrics'],
                total_seconds=round(time.time()-project['created'], 2), model_job_id=ident)
    except Exception as exc:
        _update(directory, project, '任务失败', 'failed', error=str(exc))
    finally:
        with studio.lock:
            if studio.active == ident:
                studio.active = None

def handle_get(handler, studio):
    path = urlsplit(handler.path).path
    if path == '/api/wardrobe/capabilities':
        try:
            provider = _provider()
            reason = '' if provider else '尚未配置并验证二维试衣模型；可先导入真实试衣结果再进行三维重建。'
        except Exception as exc:
            provider = None
            reason = str(exc)
        handler.reply({'reconstruction': 'LHM++ / LHMPP-700M-SMPLX-FREE', 'max_views': 4,
                       'auto_tryon_ready': bool(provider), 'tryon_provider': provider.get('name') if provider else None,
                       'tryon_reason': reason, 'text_conditioning': bool(provider and provider.get('supports_text')),
                       'active': studio.active, 'service_scope': '仅本机访问', 'research_date': '2026-11-19'})
        return True
    if path == '/api/wardrobe/projects':
        projects = []
        for file in PROJECTS.glob('*/project.json'):
            try:
                project = json.loads(file.read_text(encoding='utf8'))
                project['interrupted'] = project['status']=='running' and project['id']!=studio.active
                projects.append(project)
            except (OSError, ValueError):
                continue
        handler.reply({'active': studio.active, 'projects': sorted(projects, key=lambda p:p['created'], reverse=True)[:100]})
        return True
    return False

def handle_post(handler, studio):
    if urlsplit(handler.path).path != '/api/wardrobe/projects':
        return False
    ident = None
    directory = None
    dispatched = False
    try:
        if handler.headers.get('Origin') not in {None, 'http://127.0.0.1:8770', 'http://localhost:8770'}:
            handler.reply({'error':'Origin denied'},403)
            return True
        length = int(handler.headers.get('Content-Length', 0))
        if not 0 < length <= 220*1024**2:
            raise ValueError('请求为空或超过 220MB')
        payload = json.loads(handler.rfile.read(length))
        mode = payload.get('mode', 'reconstruct')
        if mode not in MODES:
            raise ValueError('未知任务模式')
        people = payload.get('people', [])
        if not isinstance(people, list) or not 1 <= len(people) <= 4:
            raise ValueError('请上传同一人物的 1–4 张照片')
        edited = payload.get('edited', [])
        if mode == 'import_tryon' and (not isinstance(edited, list) or len(edited) != len(people)):
            raise ValueError('真实试衣结果数量必须与人物照片相同，并按相同视角顺序上传')
        if mode != 'reconstruct' and not payload.get('garment'):
            raise ValueError('换装项目需要一张服装参考图片')
        category = payload.get('category', 'tops')
        if category not in {'tops','bottoms','one-pieces'}:
            raise ValueError('请选择有效衣服类别')
        seed = int(payload.get('seed', 42))
        if not 0 <= seed <= 2147483647:
            raise ValueError('随机种子须在 0 至 2147483647 之间')
        provider = _provider() if mode == 'auto_tryon' else None
        if mode == 'auto_tryon' and not provider:
            handler.reply({'error':'自动试衣尚未部署验证。请选择原始重建或导入真实试衣结果。'}, 503)
            return True
        with studio.lock:
            if studio.active:
                handler.reply({'error':'GPU 正在执行其他任务，请稍后提交。'}, 409)
                return True
            ident = uuid.uuid4().hex
            studio.active = ident
        directory = PROJECTS/ident
        directory.mkdir(parents=True)
        project = {'id':ident, 'name':str(payload.get('name') or '我的三维衣橱')[:70], 'mode':mode,
                   'created':time.time(), 'category':category, 'seed':seed, 'prompt':str(payload.get('prompt',''))[:1500],
                   'tryon_provider': provider.get('name') if provider else ('用户导入外部真实结果' if mode=='import_tryon' else None),
                   'text_used_by_model': bool(mode=='auto_tryon' and provider.get('supports_text')),
                   'consistency_warning':'逐视角二维试衣不保证三维一致；未观测背面由模型推测。'}
        project['people'] = [_save_image(item, directory/'people'/f'{i:02d}.png') for i,item in enumerate(people)]
        if payload.get('garment'):
            project['garment'] = _save_image(payload['garment'], directory/'garment.png')
        if mode == 'import_tryon':
            project['edited'] = [_save_image(item,directory/'edited'/f'{i:02d}.png') for i,item in enumerate(edited)]
        (directory/'prompt.txt').write_text(project['prompt'],encoding='utf8')
        _update(directory, project, '已验证输入，等待模型')
        threading.Thread(target=_worker,args=(studio,directory,project,provider),daemon=True).start()
        dispatched = True
        handler.reply(project,202)
    except Exception as exc:
        if ident and not dispatched:
            with studio.lock:
                if studio.active == ident:
                    studio.active = None
            if directory and directory.exists():
                shutil.rmtree(directory)
        handler.reply({'error':str(exc)},400)
    return True
