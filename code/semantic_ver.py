examples = [

{
"question": "Show all customers from Brazil",
"query": """
MATCH (c:Customer)
WHERE c.nationality = 'BR'
RETURN c.name, c.email
"""
},
{
"question": "Show all customers from Brazil and United Kingdom",
"query": """
MATCH (c:Customer)
WHERE c.nationality = 'BR' OR c.nationality = 'UK'
RETURN c.name, c.email
"""
},

{
"question": "List all business customers with pending KYC status",
"query": """
MATCH (c:Customer)
WHERE c.type = 'corporate'
AND c.kyc_status = 'pending'
RETURN c.name, c.email, c.kyc_status
"""
},
{
"question": "List all business customers with non rejected KYC status",
"query": """
MATCH (c:Customer)
WHERE c.type = 'corporate'
AND c.kyc_status <> 'rejected'
RETURN c.name, c.email, c.kyc_status
"""
},

{
"question": "Find all accounts belonging to nicolas silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:HAS_ACCOUNT]->(a:Account)
RETURN a
"""
},
{
"question": "Show me complete info on Nicolas Silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})
RETURN c.id, c.name, c.email, c.nationality, c.kyc_status, c.risk_level, c.type, c.date_of_birth, c.phone, c.address, c.tax_id, c.registration_date
"""
},
{
"question": "Show transactions made by Nicolas Silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
RETURN t
"""
},

{
"question": "Show transactions made by Nicolas Silva in 2023",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE t.date >= datetime('2023-01-01')
AND t.date <= datetime('2023-12-31')
RETURN t
"""
},

{
"question": "Show transactions made by Brazilian customers",
"query": """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE c.nationality = 'BR'
RETURN c.name, t
"""
},

{
"question": "Show transactions made by Brazilian business customers",
"query": """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE c.nationality = 'BR'
AND c.type = 'corporate'
RETURN c.name, t
"""
},

{
"question": "Show transactions made by Brazilian business customers with pending KYC",
"query": """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE c.nationality = 'BR'
AND c.type = 'corporate'
AND c.kyc_status = 'pending'
RETURN c.name, c.email, t
"""
},

{
"question": "Show transactions made by Brazilian business customers with pending KYC in the last 7 days",
"query": """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE c.nationality = 'BR'
AND c.type = 'corporate'
AND c.kyc_status = 'pending'
AND t.date >= datetime() - duration('P7D')
RETURN c.name, c.email, t
"""
},
{
"question": "How many british customers do we have?",
"query": """
MATCH (c:Customer)
WHERE c.nationality = 'UK'
RETURN COUNT(DISTINCT c) AS customer_count
"""
},
{
"question": "Find customers who share the same address as Nicolas Silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:LIVES_AT]->(a:Address)<-[:LIVES_AT]-(other:Customer)
RETURN other.name, other.email
"""
},

{
"question": "Find customers that share bank accounts with Nicolas Silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:HAS_ACCOUNT]->(a:Account)<-[:HAS_ACCOUNT]-(other:Customer)
WHERE other.name <> c.name
RETURN other.name, other.email
"""
},

{
"question": "Find customers that transacted with the same counterparties as Nicolas Silva",
"query": """
MATCH (c:Customer {{name:'Nicolas Silva'}})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED]->(r:Account)<-[:HAS_ACCOUNT]-(other:Customer)
WHERE other.name <> c.name
RETURN DISTINCT other.name, other.email
"""
},

{
"question": "Show customers that appear on sanctions lists",
"query": """
MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction)
RETURN c.name, s
"""
},

{
"question": "Show business customers that appear on sanctions lists",
"query": """
MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction)
WHERE c.type = 'corporate'
RETURN c.name, c.email, s
"""
},

{
"question": "Show customers that are politically exposed persons",
"query": """
MATCH (c:Customer)-[:IN_PEP]->(p:PEP)
RETURN c.name, p
"""
},

{
"question": "Show high risk customers",
"query": """
MATCH (c:Customer)
WHERE c.risk_level = 'high'
RETURN c.name, c.email, c.risk_level
"""
},

{
"question": "Show high risk customers with sanctions",
"query": """
MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction)
WHERE c.risk_level = 'high'
RETURN c.name, c.email, s
"""
},

{
"question": "Show high risk customers with transactions in the last week",
"query": """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
WHERE c.risk_level = 'high'
AND t.date >= datetime() - duration('P7D')
RETURN c.name, c.email, t
"""
}

]
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_neo4j import GraphCypherQAChain
from langchain_ollama import ChatOllama
load_dotenv()
login(os.environ["HF_TOKEN"])
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, #empty
    HuggingFaceEmbeddings(),
    Neo4jVector,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database="kyc",
    k=3,
    input_keys=["question"],
)
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries. Don't add any preambles, just return the correct cypher query. Remember cypher queries always end with a RETURN statement.\n\n",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question", "schema"],
)
assistant_prompt = PromptTemplate(
    template="""
    User question: {question}
    Context: {context}
    You are a helpful banking KYC assistant. Use the above context from the Cypher query result to answer the user's question.
    If the Cypher query returns an empty result, say \"No results found\".
    Always structure your response in the following guideline:
    1. Summary: Provide a concise summary/table/list of the result in plain language.
    2. Explanation: Explain what the number or data means for the user, a financial professional, in the context of KYC and AML compliance.
    3. Key Observations: Include up to three key observations or insights from the context. If limited data, note patterns or implications.
    4. Follow-up: Suggest one or two ways  you could help next, given its original question and the context.
    """,
    input_variables=["question", "context"],
)


print("Creating chain...")

def create_chain(model,query_model, graph):
    chain_with_dynamic_few_shot = GraphCypherQAChain.from_llm(graph=graph,
                                                          llm=query_model,
                                                          qa_prompt=None,  # Remove qa_prompt since it's not used
                                                          cypher_prompt=dynamic_prompt,
                                                          return_intermediate_steps=True,
                                                          input_key="query",
                                                          verbose=True,
                                                          validate_cypher = True,
                                                          allow_dangerous_requests=True,
                                                          return_direct=True,
                                                          use_function_response=True)
    
    class CustomQAChain:
        def __init__(self, cypher_chain, qa_model, qa_prompt):
            self.cypher_chain = cypher_chain
            self.qa_model = qa_model
            self.qa_prompt = qa_prompt
        
        def invoke(self, inputs):
            # Run the cypher chain
            cypher_result = self.cypher_chain.invoke(inputs)
            
            # Extract question and context
            question = inputs.get("query", "")
            context = cypher_result.get("intermediate_steps", [])
            if context and isinstance(context, list) and len(context) > 1:
                context = context[1]  # Usually the second step has the context
                if isinstance(context, dict) and "context" in context:
                    context = context["context"]
            else:
                context = ""
            
            # Format the prompt
            formatted_prompt = self.qa_prompt.format(question=question, context=str(context))
            # Run QA with model
            qa_result = self.qa_model.invoke([HumanMessage(content=formatted_prompt)])

            # Return combined result
            return {
                "result": qa_result.content,
                "intermediate_steps": cypher_result.get("intermediate_steps", [])
            }
    
    return CustomQAChain(chain_with_dynamic_few_shot, model, assistant_prompt)
