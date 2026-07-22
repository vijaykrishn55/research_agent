import json
with open('.research_agent/meta.json', encoding='utf-8') as f:
    meta = json.load(f)
ai_chunks = [c for c in meta['chunks'] if 'artificial_intelligence' in c['source']]
for c in ai_chunks:
    cid = c['chunk_id']
    preview = c['content'][:300].replace('\n', ' ')
    print(f"[{cid}] {preview}")
    print()
