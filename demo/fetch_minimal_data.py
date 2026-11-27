"""Extract three examples from the author's public ZIP using HTTP ranges."""
from pathlib import Path
import io
import requests
import zipfile

root = Path(__file__).resolve().parents[1]
session = requests.Session()
session.trust_env = False
base = 'https://hiteducn0-my.sharepoint.com'
session.get(base + '/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd', timeout=40).raise_for_status()
url = base + '/personal/sawyer0503_hit_edu_cn/_layouts/15/download.aspx'
params = {'SourceUrl': '/personal/sawyer0503_hit_edu_cn/Documents/GPS_Gaussian_data_ckpt/real_data.zip'}

class RemoteZip(io.RawIOBase):
    size = 2355539439
    def __init__(self): self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b''
        a, b = self.pos, self.pos + n - 1
        for attempt in range(3):
            try:
                r = session.get(url, params=params, headers={'Range': f'bytes={a}-{b}'}, timeout=(30, 90))
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != n:
                    raise IOError('Server did not return the requested range')
                self.pos += n
                return r.content
            except requests.RequestException:
                if attempt == 2: raise

with zipfile.ZipFile(RemoteZip()) as z:
    names = z.namelist()
    subjects = sorted({n.split('/')[2] for n in names if n.startswith('real_data/img/') and len(n.split('/')) > 3})[:3]
    print('Selected subjects:', subjects, flush=True)
    for info in z.infolist():
        parts = info.filename.split('/')
        if len(parts) != 4 or parts[1] not in ('img', 'mask', 'parm') or parts[2] not in subjects or not parts[3]: continue
        name = parts[3]
        if not (name in ('0.jpg', '1.jpg', '0.png', '1.png') or name in ('0_intrinsic.npy','1_intrinsic.npy','0_extrinsic.npy','1_extrinsic.npy')): continue
        target = root / 'vendor' / 'minimal_data' / Path(*parts[1:])
        if target.exists() and target.stat().st_size == info.file_size: continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(info))
        print(info.filename, info.file_size, flush=True)
print('Minimal dataset ready', flush=True)
