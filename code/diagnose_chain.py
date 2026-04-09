import inspect
import langchain_neo4j as nn
print('MODULE', nn)
print('SIG', inspect.signature(nn.GraphCypherQAChain.from_llm))
doc = nn.GraphCypherQAChain.from_llm.__doc__
print('DOC', doc[:1000] if doc else 'no doc')
