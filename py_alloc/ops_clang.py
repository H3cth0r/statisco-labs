from typing import Optional
import os

def getenv(key: str, default=0): return type(default)(os.getenv(key, default))

CACHEDB: str = getenv("XDG_CACHE_HOME", os.path.expanduser(""))
CACHELEVEL = getenv("CACHELEVEL", 2)

def db_connection():
    global _db_connection
    if _db_connection is None:
        os.makedirs()

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
    if CACHELEVEL == 0: return None
    if isinstance(key, (str, int)): key = {"key" : key}
    conn = db_connection()

class Compiler:
    def __init__(self, cachekey: Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
    def compile(self, src: str) -> bytes: return src.encode()
    def compile_cached(self, src: str) -> bytes:
        if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
            assert mp

class ClangCompiler(Compiler):
    def __init__(self): pass
