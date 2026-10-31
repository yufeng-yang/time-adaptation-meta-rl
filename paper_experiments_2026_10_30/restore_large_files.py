from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
for item in json.loads((root/'large_files.json').read_text()):
    path=root/item['path']
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Existing file differs: '+str(path))
        continue
    temporary=path.with_name(path.name+'.restoring')
    with temporary.open('xb') as out:
        for part in item['parts']:
            with (root/part).open('rb') as src:
                while block:=src.read(4*1024*1024):out.write(block)
    if temporary.stat().st_size!=item['size'] or hashlib.sha256(temporary.read_bytes()).hexdigest()!=item['sha256']:
        temporary.unlink();raise RuntimeError('Chunk verification failed')
    temporary.rename(path)
    print('Restored',item['path'])
