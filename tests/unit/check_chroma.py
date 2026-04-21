import chromadb

c = chromadb.HttpClient(host='127.0.0.1', port=9001)
col = c.get_collection('policy_notices')

for s in ['USTR', 'CBP', 'USITC', 'ITA', 'EOP']:
    docs = col.get(where={"source": {"$eq": s}})
    print(f'{s}: {len(docs["ids"])} chunks')

all_docs = col.get()
print(f'Total: {len(all_docs["ids"])} chunks')