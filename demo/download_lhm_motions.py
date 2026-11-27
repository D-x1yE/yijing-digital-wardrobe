"""Fetch the author's LHM++ motion presets with SHA-256 verification."""
from pathlib import Path
import hashlib,tarfile,requests
ROOT=Path(__file__).resolve().parent.parent/'vendor/LHM-plusplus-main'
SHA256='e2612a18c0def13cf1834951155ac8d9c28529a9bd1ac6906fe1462160c4c8bc'
ARCHIVE=ROOT/'.cache/motion_assets/motion_video.tar'
if __name__=='__main__':
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    if not ARCHIVE.exists():
        response=requests.get('https://modelscope.cn/models/Damo_XR_Lab/LHMPP-Assets/resolve/master/motion_video.tar',stream=True,timeout=120)
        response.raise_for_status()
        with ARCHIVE.with_suffix('.part').open('wb') as f:
            for chunk in response.iter_content(2**20):f.write(chunk)
        ARCHIVE.with_suffix('.part').replace(ARCHIVE)
    digest=hashlib.sha256()
    with ARCHIVE.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):digest.update(chunk)
    if digest.hexdigest()!=SHA256:raise RuntimeError('Motion archive checksum mismatch')
    root=ROOT.resolve()
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            if not (root/member.name).resolve().is_relative_to(root) or member.issym() or member.islnk():
                raise RuntimeError('Unsafe motion archive member')
        archive.extractall(root)
    print('Official motion presets ready.')
