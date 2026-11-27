"""Download public author-provided GPS-Gaussian evaluation data."""
from pathlib import Path
import requests
import time

root = Path(__file__).resolve().parents[1]
out = root / 'vendor' / 'real_data.zip'
s = requests.Session()
s.trust_env = False
share = 'https://hiteducn0-my.sharepoint.com/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd'
s.get(share, timeout=40).raise_for_status()
url = "https://hiteducn0-my.sharepoint.com/personal/sawyer0503_hit_edu_cn/_api/web/GetFileByServerRelativeUrl('/personal/sawyer0503_hit_edu_cn/Documents/GPS_Gaussian_data_ckpt/real_data.zip')/$value"
offset = out.stat().st_size if out.exists() else 0
headers = {'Range': f'bytes={offset}-'} if offset else {}
with s.get(url, headers=headers, timeout=(30, 90), stream=True) as r:
    r.raise_for_status()
    mode = 'ab' if offset and r.status_code == 206 else 'wb'
    if mode == 'wb': offset = 0
    last = time.monotonic()
    with out.open(mode) as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
            offset += len(chunk)
            if time.monotonic() - last > 10:
                print(f'{offset / 1024**2:.1f} MiB downloaded', flush=True)
                last = time.monotonic()
print('Saved', out, offset, flush=True)
